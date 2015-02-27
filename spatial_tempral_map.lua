-- the goal here is to unit test the mapping between
-- nn.TemporalConvolution and nn.SpatialConvolution

require 'nn'

dofile 'size.lua'

debug_mode = false
show_reshaped_x = debug_mode

function temporal1d(width, inp, outp, kw, dw, weight, bias)
   model = nn.Sequential()
   model:add(nn.Reshape(width, inp))
   conv = nn.TemporalConvolution(inp, outp, kw, dw)
   model:add(conv)
   
   if debug_mode then
      -- print('temporal weight')
      -- print(#conv.weight)
      -- print('temporal bias')
      -- print(#conv.bias)
      
      old_weight_size = #conv.weight
      old_bias_size = #conv.bias
   end
   
   -- update weight and bias tensors with known random values
   conv.weight = weight
   conv.bias = bias
   
   if debug_mode then
      print('temporal new weight')
      print(conv.weight)
      print('temporal new bias')
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
   
   return model
end

function spatial1d(width, inp, outp, kw, dw, weight, bias)
   height=1
   kh=1
   dh=1

   model = nn.Sequential()
   if inp ~= 1 then
      model:add(nn.Transpose({1,2}))
   end
   model:add(nn.Reshape(inp, height, width))

   if show_reshaped_x then
      print('reshaped x')
      print(model:forward(x))
   end

   conv = nn.SpatialConvolution(inp, outp, kw, kh, dw, dh)

   if debug_mode then
      -- print('conv weight')
      -- print(#conv.weight)
      -- print('conv bias')
      -- print(#conv.bias)
      
      old_weight_size = #conv.weight
      old_bias_size = #conv.bias
   end

   conv.weight = weight:resize(outp, inp, kh, kw)
   conv.bias = bias

   if debug_mode then
      print('conv new weight')
      print(conv.weight)
      print('conv new bias')
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

   model:add(conv)
   owidth = (width - kw) / dw + 1
   model:add(nn.View(outp, owidth))
   model:add(nn.Transpose({1,2}))

   return model
end

function assert_equal(width, inp, outp, kw, dw, weight, bias)
   -- build random tensors
   if inp == 1 then
      -- a special case such as for the first layer where the 1d input
      -- data hasnt yet been through any convolutions
      x = torch.rand(width)
   else
      x = torch.rand(width, inp)
   end
   weight = torch.rand(outp, (inp * kw))
   bias = torch.rand(outp)

   if debug_mode then
      print('x')
      print(x)
   end
   
   -- forward temporal1d
   model = temporal1d(width, inp, outp, kw, dw, weight, bias)
   yt = model:forward(x)
   if debug_mode then
      print('temporal forward')
      print(yt)
   end

   -- forward spatial1d
   model = spatial1d(width, inp, outp, kw, dw, weight, bias)
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
      width = 2, inp = 5, outp = 3, kw = 2, dw = 1,
   }, {
      width = 3, inp = 4, outp = 1, kw = 2, dw = 1,
   }, {
      width = 8, inp = 1, outp = 1, kw = 2, dw = 1,
   }, {
      width = 8, inp = 5, outp = 3, kw = 1, dw = 1,
   }, {
      width = 8, inp = 5, outp = 1, kw = 1, dw = 1,
   }, {
      width = 8, inp = 1, outp = 3, kw = 2, dw = 1,
   }, {
      width = 8, inp = 1, outp = 3, kw = 1, dw = 1,
   }, {
      width = 8, inp = 1, outp = 1, kw = 1, dw = 1,
   }
}
local rets = {}
for _, test in pairs(tests) do
   rets[#rets + 1] = assert_equal(test.width, test.inp, test.outp, test.kw, test.dw)
end

for i = 1, #rets do
   -- print(tests[i])
   -- print(rets[i])
   
   if rets[i] == false then
      debug_mode = true
      show_reshaped_x = debug_mode
      
      test = tests[i]
      assert_equal(test.width, test.inp, test.outp, test.kw, test.dw)
   end
end

print(rets)
