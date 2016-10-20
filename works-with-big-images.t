local qs = require('qs')
local std = require('qs.lib.std')
local distrib = require('qs.distrib')
local cmath = terralib.includec('math.h')
local torch_image = require('image')
local torch = require('torch')
local math = require('math')
local max = math.max
local min = math.min
local mod = math.mod
local abs = math.abs
local PI = 3.1415

-- config
local IMAGE_WIDTH = 338
local IMAGE_HEIGHT = 600

local RESIZED_WIDTH = 56
local RESIZED_HEIGHT = 100

local QUICK_TEST = false
--

local print_table = function(name, table)
  for k,v in pairs(table) do
    print(name, k, v)
  end
end

function interpolate(start, finish, percentage)
  return math.floor(start + (percentage * (finish-start)))
end

function fresh_image (width, height)
  return torch.Tensor(1, height, width)
end

function load_image(filename)
  return torch_image.load(filename, 1, 'double')
end

function get_edges(points)
  -- print('\nGET_EDGES')
  local edges = {}

  for i=1, #points do
    local start = points[i]

    local j = i + 1; 
      if j > #points then j = 1 end

    local finish = points[j]

    if start.y == finish.y then
      -- do nothing
    elseif start.y > finish.y then
      table.insert(edges, {finish, start})
    else
      table.insert(edges, {start, finish})
    end
  end

  return edges
end

function get_active_edges(edges, row)
  -- print('\nGET ACTIVE EDGES')
  local active_edges = {}

  for i=1, #edges do
    local edge = edges[i]

    if edge[1].y <= row and edge[2].y >= row then
      table.insert(active_edges, edges[i])
    end
  end


  return active_edges
end

function get_bounds(edges, row)
  -- print('\nGET_BOUNDS')
  local bounds = {}
  for i = 1, #edges do
    local edge = edges[i]
    local percent_done = (row - edge[1].y) / (edge[2].y - edge[1].y)
    table.insert(bounds, interpolate(edge[1].x, edge[2].x, percent_done))
  end

  return bounds
end

function render_quad(image, top_left, top_right, bottom_right, bottom_left, rgb)
  -- print('\nRENDER_QUAD')
  local image_height = image:size()[2]
  local image_width = image:size()[3]

  local edges = get_edges({top_left, top_right, bottom_right, bottom_left})

  local lowest_y = min(top_left.y, top_right.y)
  local highest_y = max(bottom_left.y, bottom_right.y)
  local range = highest_y - lowest_y

  for i = 1, range do
    local row = i + lowest_y

    local active_edges = get_active_edges(edges, row)
    local bounds = get_bounds(active_edges, row)

    -- print('bounds', bounds[1], bounds[2])

    local bound1 = bounds[1]
    local bound2 = bounds[2]

    if not bound1 then
      bound1 = 150
    end

    if not bound2 then bound2 = 150 end

    local left_bound = min(bound1, bound2)
    local right_bound = max(bound1, bound2)

    if row < 1 then row = 1 end
    if left_bound < 1 then left_bound = 1 end
    if right_bound < 1 then right_bound = 1 end

    if row > image_height then 
      row = image_height 
    end
    if left_bound > image_width then left_bound = image_width end
    if right_bound > image_width then right_bound = image_width end


    image[{1, row, {left_bound, right_bound}}] = rgb[1]

    
  end

  return image
end

local init_x_center = RESIZED_HEIGHT/2.0
local init_y_center = RESIZED_WIDTH/2.0
local init_width = 30.0
local init_floor_intensity = .3
local init_ceiling_intensity = .5
local init_wall_intensity = .7
local init_square_intensity = .8
local init_rotation = 0
local init_intensity = .5

function infer_scene(filename)
  local p2 = qs.program(function()
    qs.initrand()

    -- local uniform_discrete = qs.func(terra(low: qs.real, high: qs.real)
    --   var x = cmath.floor(qs.uniform(low, high))
    --   return x
    -- end)
    
    local target_image = torch_image.scale(load_image(filename), RESIZED_HEIGHT)

    -- some of the predicted image will be black
    -- we want to ignore those portions
    -- take all the indexes where the predicted image is zero
    -- set that to zero after the subtraction
    local compare_to_target = function(predicted_image)
  		local places_to_zero = torch.le(predicted_image, 0)
  		local white_places = torch.ge(predicted_image, .99)

  		torch.eq(torch.zeros(predicted_image:size()), predicted_image):double()
      predicted_image:csub(target_image)
  		predicted_image[places_to_zero] = torch.mean(predicted_image)
  		predicted_image[white_places] = torch.mul(predicted_image[white_places], 10)
      return torch.sum(predicted_image:pow(2))
    end

    local best_difference = 99999

    local render_image = function(x_center, y_center, width, height, floor_intensity, ceiling_intensity, wall_intensity, square_intensity, image_width, image_height)
    	local half_square_width = math.floor(width/2)
      local half_square_height = math.floor(height/2)

      local image_top_left = {x=1, y=1}
      local image_top_right = {x=image_width, y=1}
      local image_bottom_right = {x=image_width, y=image_height}
      local image_bottom_left = {x=1, y=image_height}

      local square_top_left     = {x=max(1, x_center - half_square_width), y=max(1, y_center - half_square_height)}
      local square_top_right    = {x=min(image_width, x_center + half_square_width), y=max(1, y_center - half_square_height)}
      local square_bottom_right = {x=min(image_width, x_center + half_square_width), y=min(image_height, y_center + half_square_height)}
      local square_bottom_left  = {x=max(1, x_center - half_square_width), y=min(image_height, y_center + half_square_height)}

      local image = fresh_image(image_width, image_height)

      -- -- print('\nfloor')
      image = render_quad(image, square_bottom_left, square_bottom_right, image_bottom_right, image_bottom_left, {floor_intensity, floor_intensity, floor_intensity})

      -- -- print('\nleft_wall')
      image = render_quad(image, image_top_left, square_top_left, square_bottom_left, image_bottom_left, {wall_intensity, wall_intensity, wall_intensity})

      -- -- print('\nceiling')
      image = render_quad(image, image_top_left, image_top_right, square_top_right, square_top_left, {ceiling_intensity, ceiling_intensity, ceiling_intensity})

      -- -- print('\nright_wall')
      image = render_quad(image, square_top_right, image_top_right, image_bottom_right, square_bottom_right, {wall_intensity, wall_intensity, wall_intensity})

      -- print('\nsquare')
      image = render_quad(image, square_top_left, square_top_right, square_bottom_right, square_bottom_left, {square_intensity, square_intensity, square_intensity})

      return image
  	end

    local get_image_difference = function (rotation, x_center, y_center, width, height, floor_intensity, ceiling_intensity, wall_intensity, square_intensity)

      x_center, y_center, width, height = math.floor(x_center), math.floor(y_center), math.floor(width), math.floor(height)

      local predicted_image = render_image(x_center, y_center, width, height, floor_intensity, ceiling_intensity, wall_intensity, square_intensity, RESIZED_WIDTH, RESIZED_HEIGHT)


      local hi_res_prediction = render_image(x_center * IMAGE_WIDTH  / RESIZED_WIDTH, y_center * IMAGE_HEIGHT / RESIZED_HEIGHT, width    * IMAGE_WIDTH  / RESIZED_WIDTH, height   * IMAGE_HEIGHT / RESIZED_HEIGHT, floor_intensity, ceiling_intensity, wall_intensity, square_intensity, IMAGE_WIDTH, IMAGE_HEIGHT)

      local difference = compare_to_target(torch_image.rotate(predicted_image, rotation))

      if difference < best_difference then
        best_difference = difference
        print('best difference: ', difference)
        torch_image.save(filename ..'-best.png', torch_image.rotate(hi_res_prediction, rotation))

        init_x_center = x_center 
        init_y_center = y_center 
        init_width = width 
        init_floor_intensity = floor_intensity 
        init_ceiling_intensity = ceiling_intensity 
        init_wall_intensity = wall_intensity 
        init_square_intensity = square_intensity 
        init_rotation = rotation
      end

      return difference
    end 

    local terra_get_image_diff = terralib.cast({float, float, float, float, float, float, float, float, float} -> float , get_image_difference)

    return terra()

      -- var x_center = qs.uniform(15, RESIZED_WIDTH-15)
      -- var y_center = qs.uniform(1, RESIZED_HEIGHT-20)

      -- var width = qs.uniform(RESIZED_WIDTH/6.0, RESIZED_WIDTH/2.0)
      -- var height = 1.2*width

      var width = 10
      var height = 1.2*width

      var x_center = qs.uniform(1, RESIZED_WIDTH/10) * 10
      var y_center = qs.uniform(1, RESIZED_HEIGHT/20) * 20

      x_center = cmath.fmax(width, x_center)
      y_center = cmath.fmax(height, y_center)

      x_center = cmath.fmin(RESIZED_WIDTH - width, x_center)
      y_center = cmath.fmin(RESIZED_HEIGHT - height, y_center)

      if not (width > 0) then
        print('nan problem')
        return 0
      end

      var intensity = 1

      var floor_intensity = init_floor_intensity * intensity
      var ceiling_intensity = init_ceiling_intensity * intensity
      var wall_intensity = init_wall_intensity * intensity
      var square_intensity = 1

      var rotation = qs.uniform(-PI/9, PI/9)

      var difference = terra_get_image_diff(rotation, x_center, y_center, width, height, floor_intensity, ceiling_intensity, wall_intensity, square_intensity)

      -- should take into account previous difference
      qs.factor(-difference/2000)

      return 0
    end
  end)

  -- 100 samples is about 2 per second
  -- TraceMHKernel is great
  -- HARMKernel works great and fast
  -- DriftKernel is really accurate
  -- HMC breaks quickly

  local infer
  if QUICK_TEST == true then
	  infer = qs.infer(p2, qs.MAP, qs.MCMC(qs.TraceMHKernel(), {numsamps=1, verbose=true}))
  else
	  infer = qs.infer(p2, qs.MAP, qs.MCMC(qs.TraceMHKernel(), {numsamps=1500, verbose=true}))
  end

  local terra run()
    std.printf('%f', infer())
  end

  run()
end

function left_pad(x)
  if x < 10 then
    return '000' .. x
  elseif x < 100 then
    return '00' .. x
  elseif x < 1000 then
    return '0' .. x
  end
end

if QUICK_TEST == true then
	for i=1, 1 do
	  print('i', i)
	  infer_scene('data/out'..left_pad(i)..'.png')
	end
else
	for i=1, 76 do
	  print('i', i)
	  infer_scene('data/out'..left_pad(i)..'.png')
	end
end
	

--[[ IDEAS
  factor relationships among variables
]] 




-- PIPELINE FOR DATA PREPARATION

-- resize from big .mov file
-- ffmpeg -i hallway_1.mov -vf scale=150:-1 smaller.mov

-- .mov to a bunch of .pngs
-- ffmpeg -i smaller.mov -vf fps=10 out%04d.png

-- make predictions
-- terra hallway-match.t
-- wait til done
-- mv *-best.png guesses && cd guesses

-- .pngs to .mov
-- ffmpeg -i "out%04d.png" -r 10 -c:v libx264 -crf 20 -pix_fmt yuv420p 1.mov
-- ffmpeg -i "out%04d.png-best.png" -r 10 -c:v libx264 -crf 20 -pix_fmt yuv420p 2.mov
-- r: frames per second


-- horizontal combine both movies
--[[
ffmpeg -i 2crazy-1.mov -i 2-crazy2.mov -filter_complex \
'[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
-map [vid] -c:v libx264 -crf 23 -preset veryfast 2crazy-3.mov

]] 
