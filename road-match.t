local qs = require('qs')
local std = require('qs.lib.std')
local distrib = require('qs.distrib')
local cmath = terralib.includec('math.h')
local torch_image = require('image')
local torch = require('torch')
local math = require('math')
local max = math.max
local min = math.min
local ffi = require('ffi')

-- config
local TARGET_IMAGE_NAME = 'road3'
--

qs.initrand()

local print_table = function(name, table)
  for k,v in pairs(table) do
    print(name, k, v)
  end
end
--[[
channel : row : column
]]

function interpolate(start, finish, percentage, flip)
  return math.floor(start + (percentage * (finish-start)))
end

local fresh_image = function ()
  return torch.Tensor(3, 100, 100)
end

local p2 = qs.program(function()
  local load_image = function (filename)
    return torch_image.load(filename, 3, 'double')
  end

  -- local uniform_discrete = qs.func(terra(low: qs.real, high: qs.real)
  --   var x = cmath.floor(qs.uniform(low, high))
  --   return x
  -- end)

  local render_quad = function(image, top_left, top_right, bottom_right, bottom_left, rgb)
    for row = bottom_left[2], top_left[2] do
      local percent_done = row/top_left[2]
      local left_bound = interpolate(bottom_left[1], top_left[1], percent_done, false)
      local right_bound = interpolate(bottom_right[1], top_right[1], percent_done, true)

        for i=1,3 do
          image[{i, 
                -- top_left[2] - row + 1, 
                101 - row, 
                {left_bound, right_bound}}] = rgb[i]
        end
    end
    
    return image
  end

  local a = load_image(TARGET_IMAGE_NAME .. '.jpg')

  local compare_to_target = function(b)
    b:csub(a)
    return torch.sum(b:pow(2)[{{},
      {50,100},
      {}}])
  end

  local best_difference = 99999

  local get_image_difference = function (start_width, start_center, horizon_height, end_width, end_center, r1, g1, b1, r2, g2, b2, r3, g3, b3)
    start_width, start_center, horizon_height, end_width, end_center = math.floor(start_width), math.floor(start_center), math.floor(horizon_height), math.floor(end_width), math.floor(end_center)

    local top_left  = {max(1, end_center - (end_width/2)), horizon_height}
    local top_right = {min(end_center + end_width/2, 100), horizon_height}

    local bottom_left  = {max(start_center - start_width/2, 1), 1}
    local bottom_right = {min(start_center + start_width/2, 100), 1}

    local image = fresh_image()

    -- print('ground')
    image = render_quad(image, {1, horizon_height}, {100, horizon_height},{100, 1},{1, 1},{r1, g1, b1})

    -- print('road')
    image = render_quad(image, top_left, top_right, bottom_right, bottom_left, {r2, g2, b2})

    -- print('line')
    image = render_quad(image, 
      {max(1, end_center), horizon_height}, 
      {min(100, end_center), horizon_height}, 
      {min(start_center+1, 100), 1}, 
      {max(1, start_center-1), 1}, 
      {192/255, 149/255, 25/255})

    -- print('sky')
    image = render_quad(image, {1, 100}, {100, 100}, {100, horizon_height}, {1, horizon_height}, {r3, g3, b3})

    local copy = torch.Tensor(3, 100, 100)
    copy:copy(image)

    local difference = compare_to_target(image)

    if difference < best_difference then
      best_difference = difference
      print('best difference: ', difference)
      torch_image.save('best-'.. TARGET_IMAGE_NAME ..'.png', copy)
    end

    return difference
  end 

  local terra_get_image_diff = terralib.cast({float, float, float, float, float, float, float, float, float, float, float, float, float, float} -> int8 , get_image_difference)


--[[
Hallway

]]
  -- Here's our mixture model
  -- random colo
  return terra()
    var start_width = qs.uniform(50, 100, {struc=false})
    var start_center = qs.uniform(20, 80, {struc=false})

    var horizon_height = qs.uniform(40,60, {struc=false})

    var end_width = qs.uniform(1, 50, {struc=false})
    var end_center = qs.uniform(10, 90, {struc=false})

    -- ground
    var r1 = qs.uniform(0, 1, {struc=false})
    var g1 = qs.uniform(0, 1, {struc=false})
    var b1 = qs.uniform(0, 1, {struc=false})

    -- road
    var r2 =  qs.uniform(0, 1, {struc=false})
    var g2 =  qs.uniform(0, 1, {struc=false})
    var b2 =  qs.uniform(0, 1, {struc=false})

   -- sky
    var r3 = qs.uniform(0, 1, {struc=false})
    var g3 = qs.uniform(0, 1, {struc=false})
    var b3 = qs.uniform(0, 1, {struc=false})

    -- var start_width = uniform_discrete(11, 100)
    -- var start_center = uniform_discrete(10, 90)

    -- var horizon_height = uniform_discrete(40, 60)

    -- var end_width = uniform_discrete(1, 50)
    -- var end_center = uniform_discrete(1, 100)

    var difference = terra_get_image_diff(start_width, start_center, horizon_height, end_width, end_center, r1, g1, b1, r2, g2, b2, r3, g3, b3)

    -- var factor = qs.uniform(1, 1000)

    qs.factor(-difference/40)

    return 0
  end
end)

-- Query the model for the parameters with highest
--    posterior probability.
local infer = qs.infer(p2, qs.MAP, qs.MCMC(qs.TraceMHKernel(), {numsamps=10000, verbose=true}))

local terra run()
  std.printf('%f', infer())
end
run()