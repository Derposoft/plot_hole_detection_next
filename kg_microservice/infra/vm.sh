az vm create \
  --resource-group plot_hole_detection \
  --name test \
  --image stanfordcorenlp/images/stanfordcorenlp/versions/1.1.0 \
  --ssh-key-value ../azureuser.pem \
  --authentication-type ssh \
  --public-ip-sku Basic \
  --size TODO
