require('dp')
------------------------------------------------------------------------
--[[ ImageView ]]-- 
-- A View holding a tensor of images.
------------------------------------------------------------------------
local ImagePairView, parent = torch.class("dp.ImagePairView", "dp.DataView")
ImagePairView.isImagePairView = true

-- batch x pair x height x width x channels/colors
function ImagePairView:bphwc()
   if self._view == 'bphwc' then
      return nn.Identity()
   end
   return self:transpose('bphwc')
end
