import torch
import models

#model = models.se_resnet.se_resnet50(num_classes=100)
#model = models.se_resnet.se_resnet34(num_classes=100)

#model = models.bam_resnet.bam_resnet50(num_classes=100)
#model = models.bam_resnet.bam_resnet34(num_classes=100)

#model = models.cbam_resnet.cbam_resnet50(num_classes=100)
model = models.cbam_resnet.cbam_resnet34(num_classes=100)

x = torch.randn(4, 3, 32, 32)

print model
print model(x)
