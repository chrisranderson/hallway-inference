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
local floor = math.floor
local PI = 3.1415
local log = math.log

-- config
local IMAGE_WIDTH = 602
local IMAGE_HEIGHT = 1070

local RESIZED_WIDTH = 56
local RESIZED_HEIGHT = 100
-- local RESIZED_WIDTH = 338
-- local RESIZED_HEIGHT = 600

local QUICK_TEST = false
local DEBUG = false
--

local print_table = function(name, table)
  for k,v in pairs(table) do
    print(name, k, v)
  end
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

function scale_matrix(matrix, a, b)
	local min = torch.min(matrix)
	local max = torch.max(matrix)
	local c = torch.mul(torch.add(matrix, -min), (b-a))
	local d = max - min
	return torch.add(torch.mul(c, 1/d), a)
end

function add_noise(image)
	local random_values = scale_matrix(torch.randn(1, image:size()[2], image:size()[3]), -.25, .25)
	return torch.add(image, random_values)
end

function interpolate(start, finish, percentage)
	local result = math.floor(start + (percentage * (finish-start)))
  return result
end

function fresh_image (width, height)
  return torch.Tensor(1, height, width)
end

function load_image(filename)
  return torch_image.load(filename, 1, 'double')
end

function get_edges(points)
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
  local active_edges = {}

  for i=1, #edges do
    local edge = edges[i]

    if edge[1].y <= row and edge[2].y > row then
      table.insert(active_edges, edges[i])
    end
  end

  return active_edges
end

function get_bounds(edges, row)
  local bounds = {}

  for i = 1, #edges do
    local edge = edges[i]
    local percent_done = (row - edge[1].y) / (edge[2].y - edge[1].y)
    table.insert(bounds, interpolate(edge[1].x, edge[2].x, percent_done))
  end

  return bounds
end

function render_quad(image, top_left, top_right, bottom_right, bottom_left, intensity)
  local image_height = image:size()[2]
  local image_width = image:size()[3]

  local edges = get_edges({top_left, top_right, bottom_right, bottom_left})

  local lowest_y = min(top_left.y, top_right.y) - 1
  local highest_y = max(bottom_left.y, bottom_right.y) + 1
  local range = highest_y - lowest_y

  for i = 1, range do
    local row = i + lowest_y

    local active_edges = get_active_edges(edges, row)
    local bounds = get_bounds(active_edges, row)

    local bound1 = bounds[1]
    local bound2 = bounds[2]

    if not bound1 then bound1 = RESIZED_WIDTH end
    if not bound2 then bound2 = RESIZED_WIDTH end

    local left_bound = min(bound1, bound2)
    local right_bound = max(bound1, bound2)

    if row < 1 then row = 1 end
    if left_bound < 1 then left_bound = 1 end
    if right_bound < 1 then right_bound = 1 end
    if row > image_height then row = image_height end
    if left_bound > image_width then left_bound = image_width end
    if right_bound > image_width then right_bound = image_width end

    if left_bound ~= right_bound then
    	image[{1, row, {left_bound, right_bound}}] = intensity
    end
  end
end

function remove_border(image, amount_from_height)
	local height = image:size()[2]
	local width = image:size()[3]
	local x_center = floor(width/2)
	local y_center = floor(height/2)
	local amount_from_width = floor((width * ( amount_from_height)) / height)
	return torch_image.scale(torch_image.crop(image, 'c', width - amount_from_width, height - amount_from_height), max(width, height))
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
local total_difference = 999999

function infer_scene(filename)
  local p2 = qs.program(function()
    qs.initrand()

    local target_image = torch_image.scale(load_image(filename), RESIZED_HEIGHT)

    local compare_to_target = function(predicted_image)
  		-- local places_to_zero = torch.le(predicted_image, 0)
  		-- local white_places = torch.ge(predicted_image, .99)
      predicted_image:csub(target_image)
  		-- predicted_image[places_to_zero] = torch.mean(predicted_image)
  		-- predicted_image[white_places] = torch.mul(predicted_image[white_places], 10)
      return torch.sqrt(torch.sum(predicted_image:pow(2)))
    end

    local best_difference = 999999

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

      -- floor, left wall, ceiling, right wall, square
      render_quad(image, square_bottom_left, square_bottom_right, image_bottom_right, image_bottom_left, floor_intensity)
      render_quad(image, image_top_left, square_top_left, square_bottom_left, image_bottom_left, wall_intensity)
      render_quad(image, image_top_left, image_top_right, square_top_right, square_top_left, ceiling_intensity)
      render_quad(image, square_top_right, image_top_right, image_bottom_right, square_bottom_right, wall_intensity)
      render_quad(image, square_top_left, square_top_right, square_bottom_right, square_bottom_left, square_intensity)

      return image
  	end

    local get_image_difference = function (rotation, x_center, y_center, width, height, floor_intensity, ceiling_intensity, wall_intensity, square_intensity)

      x_center, y_center, width, height = math.floor(x_center), math.floor(y_center), math.floor(width), math.floor(height)

      local predicted_image = torch_image.rotate(render_image(x_center, y_center, width, height, floor_intensity, ceiling_intensity, wall_intensity, square_intensity, RESIZED_WIDTH, RESIZED_HEIGHT), rotation)
      predicted_image = add_noise(remove_border(predicted_image, 20))

      local difference = compare_to_target(predicted_image)

      if difference < best_difference then

	      total_difference = total_difference - best_difference + difference

	      if total_difference < 0 then
	      	total_difference = total_difference + 999999
	      end

        best_difference = difference

        init_x_center = x_center 
        init_y_center = y_center 
        init_width = width 
        init_floor_intensity = floor_intensity 
        init_ceiling_intensity = ceiling_intensity 
        init_wall_intensity = wall_intensity 
        init_square_intensity = square_intensity 
        init_rotation = rotation

        print('best difference, total_difference ', difference, total_difference)

        local hi_res_prediction = render_image(x_center * IMAGE_WIDTH  / RESIZED_WIDTH, y_center * IMAGE_HEIGHT / RESIZED_HEIGHT, width    * IMAGE_WIDTH  / RESIZED_WIDTH, height   * IMAGE_HEIGHT / RESIZED_HEIGHT, floor_intensity, ceiling_intensity, wall_intensity, square_intensity, IMAGE_WIDTH, IMAGE_HEIGHT)
        torch_image.save(filename ..'-best.png', add_noise(remove_border(torch_image.rotate(hi_res_prediction, rotation),  20*10)))
      end

      return difference
    end 

    local terra_get_image_diff = terralib.cast({float, float, float, float, float, float, float, float, float} -> float , get_image_difference)

    return terra()

      var box_width = qs.uniform(4, 10)
      var box_height = box_width

      var x_center = qs.uniform(1, RESIZED_WIDTH/20) * 20
      var y_center = qs.uniform(1, RESIZED_HEIGHT/40) * 40

      if not (box_width > 0) then
        print('nan problem')
        return 0
      end

      var floor_intensity = qs.uniform(.1, .8)
      var ceiling_intensity = qs.uniform(.2, .7) 
      var wall_intensity = qs.uniform(.5, 1) 
      var square_intensity = 1

      var rotation = qs.uniform((-PI/9), (PI/9))

      var difference = terra_get_image_diff(rotation, x_center, y_center, box_width, box_height, floor_intensity, ceiling_intensity, wall_intensity, square_intensity)

      qs.factor(-difference)

      return 0
    end
  end)

  local infer

  if QUICK_TEST == true then
	  infer = qs.infer(p2, qs.MAP, qs.MCMC(qs.TraceMHKernel(), {numsamps=1, verbose=true}))
  else
	  infer = qs.infer(p2, qs.MAP, qs.MCMC(qs.TraceMHKernel(), {numsamps=4000, verbose=true}))
  end

  local terra run()
    std.printf('%f', infer())
  end

  run()
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

-- slow down: ffmpeg -i 3.mov -filter:v "setpts=20.0*PTS" 3-slow.mov


-- horizontal combine both movies
--[[
ffmpeg -i 1.mov -i 2.mov -filter_complex \
'[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
-map [vid] -c:v libx264 -crf 23 -preset veryfast 3.mov
]] 
