function TemporalConvolutionMM(width, inp, outp, kw, dw)
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
