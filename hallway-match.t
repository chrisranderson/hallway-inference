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
local ffi = require('ffi')

-- config
local TARGET_IMAGE_NAME = 'out116'
local IMAGE_WIDTH = 150.0
local IMAGE_HEIGHT = 84.0
--

-- manual casting from dualnum to float


--

local print_table = function(name, table)
  for k,v in pairs(table) do
    print(name, k, v)
  end
end
--[[
channel : row : column
]]

function interpolate(start, finish, percentage)
  return math.floor(start + (percentage * (finish-start)))
end

function fresh_image ()
  return torch.Tensor(1, IMAGE_HEIGHT, IMAGE_WIDTH)
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
  local edges = get_edges({top_left, top_right, bottom_right, bottom_left})

  local lowest_y = min(top_left.y, top_right.y)
  local highest_y = max(bottom_left.y, bottom_right.y)
  local range = highest_y - lowest_y

  for i = 1, range do
    local row = i + lowest_y

    local active_edges = get_active_edges(edges, row)
    local bounds = get_bounds(active_edges, row)

    local left_bound = min(bounds[1], bounds[2])
    local right_bound = max(bounds[1], bounds[2])

    if row < 1 then row = 1 end
    if left_bound < 1 then left_bound = 1 end
    if right_bound < 1 then right_bound = 1 end

    if row > IMAGE_HEIGHT then row = IMAGE_HEIGHT end
    if left_bound > IMAGE_WIDTH then left_bound = IMAGE_WIDTH end
    if right_bound > IMAGE_WIDTH then right_bound = IMAGE_WIDTH end

    -- print('row, left_bound, right_bound', row, left_bound, right_bound)

    for color_index=1, 1 do
      image[{
        color_index,
        row,
        {left_bound, right_bound}}] = rgb[color_index]

    end
  end

  return image
end

function infer_scene(filename)
  local p2 = qs.program(function()
    qs.initrand()

    -- local uniform_discrete = qs.func(terra(low: qs.real, high: qs.real)
    --   var x = cmath.floor(qs.uniform(low, high))
    --   return x
    -- end)
    
    local a = load_image(filename)

    local compare_to_target = function(b)
      b:csub(a)
      return torch.sum(b:pow(2))
    end

    local best_difference = 99999

    local sample_number = 1
    local get_image_difference = function (x_center, y_center, width, height, floor_intensity, ceiling_intensity, wall_intensity, square_intensity)
      x_center, y_center, width, height = math.floor(x_center), math.floor(y_center), math.floor(width), math.floor(height)

      local half_square_width = math.floor(width/2)
      local half_square_height = math.floor(height/2)

      local image_top_left = {x=1, y=1}
      local image_top_right = {x=IMAGE_WIDTH, y=1}
      local image_bottom_right = {x=IMAGE_WIDTH, y=IMAGE_HEIGHT}
      local image_bottom_left = {x=1, y=IMAGE_HEIGHT}

      local square_top_left     = {x=max(1, x_center - half_square_width), y=max(1, y_center - half_square_height)}
      local square_top_right    = {x=min(IMAGE_WIDTH, x_center + half_square_width), y=max(1, y_center - half_square_height)}
      local square_bottom_right = {x=min(IMAGE_WIDTH, x_center + half_square_width), y=min(IMAGE_HEIGHT, y_center + half_square_height)}
      local square_bottom_left  = {x=max(1, x_center - half_square_width), y=min(IMAGE_HEIGHT, y_center + half_square_height)}

      local image = fresh_image()

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

      local copy = torch.Tensor(1, IMAGE_HEIGHT, IMAGE_WIDTH)

      copy:copy(image)

      local difference = compare_to_target(image)

      if difference < best_difference then
        best_difference = difference
        print('best difference: ', difference)
        torch_image.save(filename ..'-best.png', copy)
      end

      return difference
    end 

    local terra_get_image_diff = terralib.cast({float, float, float, float, float, float, float, float} -> float , get_image_difference)

    return terra()

      var x_center = qs.gaussian(IMAGE_WIDTH/2.0, 10.0, {struc=false})
      var y_center = qs.gaussian(IMAGE_HEIGHT/2.0, 10.0, {struc=false})

      var width = qs.gaussian(20.0, 5.0, {struc=false})
      var height = qs.gaussian(20.0, 5.0, {struc=false})

      var floor_intensity = qs.uniform(0.0, 1.0, {struc=false})
      var ceiling_intensity = qs.uniform(0.0, 1.0, {struc=false})
      var wall_intensity = qs.uniform(0.0, 1.0, {struc=false})
      var square_intensity = qs.uniform(0.0, 1.0, {struc=false})

      var difference = terra_get_image_diff(x_center, y_center, width, height, floor_intensity, ceiling_intensity, wall_intensity, square_intensity)

      -- should take into account previous difference
      qs.factor(-difference)

      return 0
    end
  end)

  -- 100 samples is about 2 per second
  -- TraceMHKernel is great
  -- HARMKernel works great and fast
  -- DriftKernel is really accurate
  -- HMC breaks quickly
  local infer = qs.infer(p2, qs.MAP, qs.MCMC(qs.DriftKernel(), {numsamps=50000, verbose=true}))

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


for i=217, 217 do
  infer_scene('data/out'..left_pad(i)..'.png')
end

-- run it longer
-- expectation
-- different sampling techniques

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
ffmpeg -i 1.mov -i 2.mov -filter_complex \
'[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
-map [vid] -c:v libx264 -crf 23 -preset veryfast 3.mov

]] 
