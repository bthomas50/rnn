------------------------------------------------------------------------
--[[ ScaledSpatialGlimpse ]]--
-- Ref A.: http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- Yo dawg, I heard you like glimpses so I took a glimpse and made it able to take a different scale.
------------------------------------------------------------------------
local ScaledSpatialGlimpse, parent = torch.class("nn.ScaledSpatialGlimpse", "nn.Module")

function ScaledSpatialGlimpse:__init(size, depth)
   require 'nnx'
   if torch.type(size)=='table' then
      self.height = size[1]
      self.width = size[2]
   else
      self.width = size
      self.height = size
   end
   self.depth = depth or 3
   
   assert(torch.type(self.width) == 'number')
   assert(torch.type(self.height) == 'number')
   assert(torch.type(self.depth) == 'number')
   parent.__init(self)
   self.gradInput = {torch.Tensor(), torch.Tensor()}
   self.module = nn.SpatialReSampling{oheight=self.height,owidth=self.width}
   self.modules = {self.module}
end

-- a bandwidth limited sensor which focuses on a location.
-- locations index the x,y coord of the center of the output glimpse
function ScaledSpatialGlimpse:updateOutput(inputTable)
   assert(torch.type(inputTable) == 'table')
   assert(#inputTable >= 2)
   local input, locationAndScale = unpack(inputTable)
   input = self:toBatch(input, 3)
   local location = self:toBatch(locationAndScale[{{},{1,2}}], 1)
   local scale = self:toBatch(locationAndScale[{{},{3}}], 1)
   assert(input:dim() == 4 and location:dim() == 2 and scale:dim() == 2)
   self.output:resize(input:size(1), self.depth, input:size(2), self.height, self.width)
   
   self._crop = self._crop or self.output.new()
   self._pad = self._pad or input.new()
   
   for sampleIdx=1,self.output:size(1) do
      local outputSample = self.output[sampleIdx]
      local inputSample = input[sampleIdx]
      local yx = location[sampleIdx]
      -- (-1,-1) top left corner, (1,1) bottom right corner of image
      local y, x = yx:select(1,1), yx:select(1,2)
      -- (0,0), (1,1)
      y, x = (y+1)/2, (x+1)/2
      
      -- for each depth of glimpse : pad, crop, downscale
      local glimpseWidth = self.width
      local glimpseHeight = self.height
      for depth=1,self.depth do 
         local dst = outputSample[depth]
         if depth > 1 then
            glimpseWidth = glimpseWidth*scale
            glimpseHeight = glimpseHeight*scale
         end
         
         -- add zero padding (glimpse could be partially out of bounds)
         local padWidth = math.floor((glimpseWidth-1)/2)
         local padHeight = math.floor((glimpseHeight-1)/2)
         self._pad:resize(input:size(2), input:size(3)+padHeight*2, input:size(4)+padWidth*2):zero()
         local center = self._pad:narrow(2,padHeight+1,input:size(3)):narrow(3,padWidth+1,input:size(4))
         center:copy(inputSample)
         
         -- crop it
         local h, w = self._pad:size(2)-glimpseHeight, self._pad:size(3)-glimpseWidth
         local y, x = math.min(h,math.max(0,y*h)),  math.min(w,math.max(0,x*w))
         
         if depth == 1 then
            dst:copy(self._pad:narrow(2,y+1,glimpseHeight):narrow(3,x+1,glimpseWidth))
         else
            self._crop:resize(input:size(2), glimpseHeight, glimpseWidth)
            self._crop:copy(self._pad:narrow(2,y+1,glimpseHeight):narrow(3,x+1,glimpseWidth))
         
            dst:copy(self.module:updateOutput(self._crop))
         end
      end
   end
   
   self.output:resize(input:size(1), self.depth*input:size(2), self.height, self.width)
   self.output = self:fromBatch(self.output, 1)
   return self.output
end

function ScaledSpatialGlimpse:updateGradInput(inputTable, gradOutput)
   local gradInput, gradLocation = unpack(self.gradInput)
   local input, locationAndScale = unpack(inputTable)
   input = self:toBatch(input, 3)
   local location = self:toBatch(locationAndScale[{{},{1,2}}], 1)
   local scale = self:toBatch(locationAndScale[{{},{3}}], 1)

   gradOutput = self:toBatch(gradOutput, 3)
   
   gradInput:resizeAs(input):zero()
   gradLocation:resizeAs(locationAndScale):zero() -- no backprop through location
   
   gradOutput = gradOutput:view(input:size(1), self.depth, input:size(2), self.height, self.width)
   
   for sampleIdx=1,gradOutput:size(1) do
      local gradOutputSample = gradOutput[sampleIdx]
      local gradInputSample = gradInput[sampleIdx]
      local yx = location[sampleIdx] -- height, width
      -- (-1,-1) top left corner, (1,1) bottom right corner of image
      local y, x = yx:select(1,1), yx:select(1,2)
      -- (0,0), (1,1)
      y, x = (y+1)/2, (x+1)/2
      
      -- for each depth of glimpse : pad, crop, downscale
      local glimpseWidth = self.width
      local glimpseHeight = self.height
      for depth=1,self.depth do 
         local src = gradOutputSample[depth]
         if depth > 1 then
            glimpseWidth = glimpseWidth*scale
            glimpseHeight = glimpseHeight*scale
         end
         
         -- add zero padding (glimpse could be partially out of bounds)
         local padWidth = math.floor((glimpseWidth-1)/2)
         local padHeight = math.floor((glimpseHeight-1)/2)
         self._pad:resize(input:size(2), input:size(3)+padHeight*2, input:size(4)+padWidth*2):zero()
         
         local h, w = self._pad:size(2)-glimpseHeight, self._pad:size(3)-glimpseWidth
         local y, x = math.min(h,math.max(0,y*h)),  math.min(w,math.max(0,x*w))
         local pad = self._pad:narrow(2, y+1, glimpseHeight):narrow(3, x+1, glimpseWidth)
         
         -- upscale glimpse for different depths
         if depth == 1 then
            pad:copy(src)
         else
            self._crop:resize(input:size(2), glimpseHeight, glimpseWidth)
            
            pad:copy(self.module:updateGradInput(self._crop, src))
         end
        
         -- copy into gradInput tensor (excluding padding)
         gradInputSample:add(self._pad:narrow(2, padHeight+1, input:size(3)):narrow(3, padWidth+1, input:size(4)))
      end
   end
   
   self.gradInput[1] = self:fromBatch(gradInput, 1)
   self.gradInput[2] = self:fromBatch(gradLocation, 1)
   
   return self.gradInput
end
