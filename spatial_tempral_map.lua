-- the goal here is to unit test the mapping between
-- nn.TemporalConvolution and nn.SpatialConvolution

require 'nn'

debug_mode = false

-- width of the 1d sequences to feed to the 1d convolution
width = 8

inp=2;  -- dimensionality of one sequence element
outp=5; -- number of derived features for one sequence element
kw=3;   -- kernel only operates on one sequence element per step
dw=4;   -- we step once and go on to the next sequence element


if inp == 1 then
   -- a special case such as for the first layer where the 1d input
   -- data hasnt yet been through any convolutions
   x = torch.rand(width)
else
   x = torch.rand(width, inp)
end

print('x')
print(x)

weight = torch.rand(outp, (inp * kw))
bias = torch.rand(outp)

function build_temporal_convolution(inp, outp, kw, dw, weight, bias)
   mlp_t = nn.Sequential()
   mlp_t:add(nn.Reshape(width, inp))
   mlp_t:add(nn.TemporalConvolution(inp, outp, kw, dw))

   if debug_mode then
      print('tc weight')
      print(mlp_t[2].weight)
      print('tc bias')
      print(mlp_t[2].bias)
   end
   
   -- update weight and bias tensors with known random values
   mlp_t.modules[1].weight = weight
   mlp_t.modules[1].bias = bias
   
   if debug_mode then
      print('tc new weight')
      print(mlp_t[2].weight)
      print('tc new bias')
      print(mlp_t[2].bias)
   end
   
   return mlp_t
end

mlp_t = build_temporal_convolution(inp, outp, kw, dw, weight, bias)
yt = mlp_t:forward(x)
print(yt)

kh=1
dh=1

mlp_s = nn.Sequential()
mlp_s:add(nn.Reshape(inp, 1, width))
sc = nn.SpatialConvolution(inp, outp, kw, kh, dw, dh)

if debug_mode then
   print('sc weight')
   print(sc.weight)
   print('sc bias')
   print(sc.bias)
end

sc.weight = weight:resize(outp, inp, 1, kw)
sc.bias = bias

if debug_mode then
   print('sc new weight')
   print(sc.weight)
   print('sc new bias')
   print(sc.bias)
end

mlp_s:add(sc)
owidth = (width - kw) / dw + 1
mlp_s:add(nn.View(outp, owidth))
mlp_s:add(nn.Transpose({1,2}))

ys = mlp_s:forward(x)
print(ys)

print((yt - ys):sum() < 0.000000001)
