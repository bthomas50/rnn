
--draw a box on an image
function drawBox(img, bbox, channel)
  channel = channel or 1

  local x1, y1 = torch.round(bbox[1]), torch.round(bbox[2])
  local x2, y2 = torch.round(bbox[1] + bbox[3]), torch.round(bbox[2] + bbox[4])

  x1, y1 = math.max(1, x1), math.max(1, y1)
  x2, y2 = math.min(img:size(3), x2), math.min(img:size(2), y2)

  local max = img:max()

  for i=x1,x2 do
      img[channel][y1][i] = max
      img[channel][y2][i] = max
  end
  for i=y1,y2 do
      img[channel][i][x1] = max
      img[channel][i][x2] = max
  end

  return img
end