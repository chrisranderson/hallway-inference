terralib.linklibrary('/usr/include/GL/glut.h')
local qs = require('qs')
local test = terralib.includec('blender.c')

local p2 = qs.program(function()
  qs.initrand()

  return terra()

    var x = qs.uniform(0, 1)
    qs.factor(x)

    return 0
  end
end)

test.display_everything(2, "323")

local infer = qs.infer(p2, qs.MAP, qs.MCMC(qs.TraceMHKernel(), {numsamps=1000, verbose=true}))

local terra run()
  std.printf('%f', infer())
end

run()
