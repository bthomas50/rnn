
--input x,y range from (-1,-1) top left corner, (1,1) bottom right corner of image
function locatorXYToImageXY(x, y, dataset)
  -- rescale to (0,0), (1,1)
  local imageX, imageY = (x+1)/2, (y+1)/2
  return imageX*(ds:imageSize('w')-1)+1, imageY*(ds:imageSize('h')-1)+1
end