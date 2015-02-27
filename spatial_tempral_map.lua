-- the goal here is to unit test the mapping between
-- nn.TemporalConvolution and nn.SpatialConvolution

require 'nn'

dofile 'size.lua'

debug_mode = false

function temporal1d(width, inp, outp, kw, dw)
   return nn.TemporalConvolution(inp, outp, kw, dw)
end

function temporal1d_setparams(conv, width, inp, outp, kw, dw, weight, bias)
   if debug_mode then
      old_weight_size = #conv.weight
      old_bias_size = #conv.bias
   end
   
   -- update weight and bias tensors with known random values
   conv.weight = weight
   conv.bias = bias
   
   if debug_mode then
      print('temporal weight')
      print(conv.weight)
      print('temporal bias')
      print(conv.bias)
      
      if tostring(#conv.weight) ~= tostring(old_weight_size) then
         print('WARNING: temporal weight size changed')
         print('old', old_weight_size)
         print('new', #conv.weight)
      end

      if tostring(#conv.bias) ~= tostring(old_bias_size) then
         print('WARNING: temporal bias size changed')
         print('old', old_bias_size)
         print('new', #conv.bias)
      end
   end
end

function spatial1d(width, inp, outp, kw, dw)
   height=1
   kh=1
   dh=1

   output_width = (width - kw) / dw + 1

   -- batchmode false on Reshape keeps reshape from heuristically
   -- deciding that the first dimension is a batch

   model = nn.Sequential()
   model:add(nn.Transpose({1,2}))
   model:add(nn.Reshape(inp, height, width, false))
   model:add(nn.SpatialConvolution(inp, outp, kw, kh, dw, dh))
   model:add(nn.View(outp, output_width))
   model:add(nn.Transpose({1,2}))

   return model
end

function spatial1d_debug(model)
   -- TODO: use search of some kind instead of hard coded modules[1], modules[2]
   
   if debug_mode then
      print('transposed x')
      print(model.modules[1]:forward(x))
   end

   if debug_mode then
      print('reshaped x')
      print(model.modules[2]:forward(model.modules[1]:forward(x)))
   end
end

function spatial1d_setparams(model, inp, outp, kw, dw, weight, bias)
   for i, mod in pairs(model.modules) do
      if tostring(mod) == 'nn.SpatialConvolution' then
         conv = mod
         break
      end
   end
   
   if debug_mode then
      old_weight_size = #conv.weight
      old_bias_size = #conv.bias
   end

   -- reshape splits [1,2,3,4,5,6] into [[1,2],[3,4],[5,6]] instead of
   -- [[1,4],[2,5],[3,6]] or something like that
   
   -- conv.weight = weight:reshape(outp, inp, kh, kw)
   local w = torch.Tensor(outp, inp, kh, kw)
   for o = 1, outp do
      for i = 1, inp do
         for j = 1, kw do
            -- this one is equiv to the above reshape
            -- w[o][i][1][j] = weight[o][(i-1)*kw + j]
            w[o][i][1][j] = weight[o][(j-1)*inp + i]
         end
      end
   end
   conv.weight = w
   conv.bias = bias

   if debug_mode then
      print('spatial weight')
      print(conv.weight)
      print('spatial bias')
      print(conv.bias)

      if tostring(#conv.weight) ~= tostring(old_weight_size) then
         print('WARNING: spatial weight size changed')

         print('old', old_weight_size)
         print('new', #conv.weight)
      end

      if tostring(#conv.bias) ~= tostring(old_bias_size) then
         print('WARNING: spatial bias size changed')
         
         print('old', old_bias_size)
         print('new', #conv.bias)
      end
   end
end

function assert_equal(width, inp, outp, kw, dw, weight, bias)
   -- build random tensors
   x = torch.rand(width, inp)
   weight = torch.rand(outp, (inp * kw))
   bias = torch.rand(outp)

   if debug_mode then
      print('x')
      print(x)
   end
   
   -- forward temporal1d
   model = temporal1d(width, inp, outp, kw, dw)
   temporal1d_setparams(model, width, inp, outp, kw, dw, weight, bias)
   yt = model:forward(x)
   if debug_mode then
      print('temporal forward')
      print(yt)
   end

   -- forward spatial1d
   model = spatial1d(width, inp, outp, kw, dw)
   spatial1d_setparams(model, inp, outp, kw, dw, weight, bias)
   spatial1d_debug(model)
   ys = model:forward(x)
   if debug_mode then
      print('spatial forward')
      print(ys)
   end

   -- check diff
   diff = (yt - ys):abs():sum()
   pass = diff < 0.000000001
   
   if debug_mode then
      print(diff, pass)
   end

   return pass
end

local tests = {
   {
      width = 2, inp = 3, outp = 1, kw = 2, dw = 1,
   }, {
      width = 2, inp = 1, outp = 1, kw = 2, dw = 1,
   }, {
      width = 3, inp = 1, outp = 4, kw = 2, dw = 1,
   }, {
      width = 3, inp = 3, outp = 5, kw = 3, dw = 1,
   }, {
      width = 3, inp = 5, outp = 3, kw = 2, dw = 1,
   }, {
      width = 3, inp = 4, outp = 1, kw = 2, dw = 1,
   }, {
      width = 8, inp = 1, outp = 1, kw = 2, dw = 1,
   }, {
      width = 8, inp = 5, outp = 3, kw = 1, dw = 1,
   }, {
      width = 3, inp = 2, outp = 1, kw = 1, dw = 1,
   }, {
      width = 3, inp = 1, outp = 2, kw = 1, dw = 1,
   }, {
      width = 3, inp = 1, outp = 2, kw = 1, dw = 1,
   }, {
      width = 8, inp = 1, outp = 1, kw = 1, dw = 1,
   }, {
      width = 8, inp = 1, outp = 1, kw = 1, dw = 2,
   }, {
      width = 8, inp = 1, outp = 1, kw = 2, dw = 2,
   }, {
      width = 8, inp = 1, outp = 2, kw = 2, dw = 2,
   }, {
      width = 8, inp = 2, outp = 2, kw = 2, dw = 2,
   }, {
      width = 8, inp = 2, outp = 1, kw = 2, dw = 2,
   }
}

debug_mode = true
local rets = {}
for _, test in pairs(tests) do
   print(test)
   rets[#rets + 1] = assert_equal(test.width, test.inp, test.outp, test.kw, test.dw)
end

if debug_mode == false then
   for i = 1, #rets do
      if rets[i] == false then
         debug_mode = true
         
         test = tests[i]
         print(test)
         assert_equal(test.width, test.inp, test.outp, test.kw, test.dw)
      end
   end
end

print(rets)
