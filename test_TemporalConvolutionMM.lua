-- the goal here is to unit test the mapping between
-- nn.TemporalConvolution and nn.SpatialConvolution

require 'nn'

debug_mode = false

function Temporal1D(width, inp, outp, kw, dw)
   return nn.TemporalConvolution(inp, outp, kw, dw)
end

function Temporal1D_setparams(conv, width, inp, outp, kw, dw, weight, bias)
   -- set weight and bias in a specific way, and with debugging
   -- enabled so that we can compare the forward pass with a different
   -- algorithm working on the same parameters

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

dofile 'TemporalConvolutionMM.lua'

function Spatial1D_debug(model)
   if debug_mode then
      print('transposed x')
      print(model.modules[1]:forward(x))
   end

   if debug_mode then
      print('reshaped x')
      print(model.modules[2]:forward(model.modules[1]:forward(x)))
   end
end

function Spatial1D_setparams(model, inp, outp, kw, dw, weight, bias)
   -- set weight and bias in a specific way, and with debugging
   -- enabled so that we can compare the forward pass with a different
   -- algorithm working on the same parameters

   for i, mod in pairs(model.modules) do
      if (tostring(mod) == 'nn.SpatialConvolution') or
         (tostring(mod) == 'nn.SpatialConvolutionMM') then
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

function assert_equal(bs, width, inp, outp, kw, dw, weight, bias)
   -- build random tensors
   if bs > 1 then
      x = torch.rand(bs, width, inp)
   else
      x = torch.rand(width, inp)
   end
   weight = torch.rand(outp, (inp * kw))
   bias = torch.rand(outp)

   if debug_mode then
      print('x')
      print(x)
   end
   
   -- forward Temporal1D
   model = Temporal1D(width, inp, outp, kw, dw)
   Temporal1D_setparams(model, width, inp, outp, kw, dw, weight, bias)
   yt = model:forward(x)
   if debug_mode then
      print('temporal forward')
      print(yt)
   end

   -- forward Spatial1D
   model = TemporalConvolution(bs, width, inp, outp, kw, dw)
   Spatial1D_setparams(model, inp, outp, kw, dw, weight, bias)
   Spatial1D_debug(model)
   ys = model:forward(x)
   if debug_mode then
      print('spatial forward')
      print(ys)
   end

   -- check diff
   diff = (yt - ys):abs():sum()
   pass = diff < 0.000000001

   -- check shape
   pass = pass and (tostring(#yt) == tostring(#ys))
   
   if debug_mode then
      print(#yt, #ys)
      print(diff, pass)
   end

   return pass
end

local tests = {
   {
      bs = 1, width = 2, inp = 3, outp = 1, kw = 2, dw = 1,
   }, {
      bs = 1, width = 2, inp = 1, outp = 1, kw = 2, dw = 1,
   }, {
      bs = 1, width = 3, inp = 1, outp = 4, kw = 2, dw = 1,
   }, {
      bs = 1, width = 3, inp = 3, outp = 5, kw = 3, dw = 1,
   }, {
      bs = 1, width = 3, inp = 5, outp = 3, kw = 2, dw = 1,
   }, {
      bs = 1, width = 3, inp = 4, outp = 1, kw = 2, dw = 1,
   }, {
      bs = 1, width = 8, inp = 1, outp = 1, kw = 2, dw = 1,
   }, {
      bs = 1, width = 8, inp = 5, outp = 3, kw = 1, dw = 1,
   }, {
      bs = 1, width = 3, inp = 2, outp = 1, kw = 1, dw = 1,
   }, {
      bs = 1, width = 3, inp = 1, outp = 2, kw = 1, dw = 1,
   }, {
      bs = 1, width = 3, inp = 1, outp = 2, kw = 1, dw = 1,
   }, {
      bs = 1, width = 8, inp = 1, outp = 1, kw = 1, dw = 1,
   }, {
      bs = 1, width = 8, inp = 1, outp = 1, kw = 1, dw = 2,
   }, {
      bs = 1, width = 8, inp = 1, outp = 1, kw = 2, dw = 2,
   }, {
      bs = 1, width = 8, inp = 1, outp = 2, kw = 2, dw = 2,
   }, {
      bs = 1, width = 8, inp = 2, outp = 2, kw = 2, dw = 2,
   }, {
      bs = 1, width = 8, inp = 2, outp = 1, kw = 2, dw = 2,
   }, {
      bs = 15, width = 8, inp = 2, outp = 1, kw = 2, dw = 2,
   }
}

debug_mode = true
local rets = {}
for _, test in pairs(tests) do
   print(test)
   rets[#rets + 1] = assert_equal(test.bs, test.width, test.inp, test.outp, test.kw, test.dw)
end

if debug_mode == false then
   for i = 1, #rets do
      if rets[i] == false then
         debug_mode = true
         
         test = tests[i]
         print(test)
         assert_equal(test.bs, test.width, test.inp, test.outp, test.kw, test.dw)
      end
   end
end

print(rets)
