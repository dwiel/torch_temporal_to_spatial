require 'nn'

function _SpatialTemporalConvolution(bs, width, inp, outp, kw, dw, SpatialConvolution)
   -- make sure that SpatialConvolution is one of the two valid ones
   assert(
      (SpatialConvolution == nn.SpatialConvolution) or
      (SpatialConvolution == nn.SpatialConvolutionMM)
   )

   height=1
   kh=1
   dh=1

   output_width = (width - kw) / dw + 1

   -- batchmode false on Reshape keeps reshape from heuristically
   -- deciding that the first dimension is a batch

   model = nn.Sequential()
   if bs == 1 then
      model:add(nn.Transpose({1,2}))
      model:add(nn.Reshape(inp, height, width, false))
   else
      model:add(nn.Transpose({2,3}))
      model:add(nn.Reshape(bs, inp, height, width, false))
   end
   model:add(SpatialConvolution(inp, outp, kw, kh, dw, dh))
   if bs == 1 then
      -- somehow, setting NumInputDims to 1 gets the output to be size
      -- 2 ... I think it is treating the first dim as a batch, or
      -- something
      model:add(nn.View(outp, output_width):setNumInputDims(1))
      model:add(nn.Transpose({1,2}))
   else
      model:add(nn.View(bs, outp, output_width))
      model:add(nn.Transpose({2,3}))
   end

   return model
end

function TemporalConvolutionMM(bs, width, inp, outp, kw, dw)
   -- use nn.SpatialConvolutionMM
   return _SpatialTemporalConvolution(bs, width, inp, outp, kw, dw, nn.SpatialConvolutionMM)
end

function TemporalConvolution(bs, width, inp, outp, kw, dw)
   -- use nn.SpatialConvolution
   return _SpatialTemporalConvolution(bs, width, inp, outp, kw, dw, nn.SpatialConvolution)
end
