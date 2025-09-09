import torch
from torch import nn
from torchmetrics import Accuracy

from module.gradient_reverse_function import WarmStartGradientReverseLayer

    
class DomainAdversarialLoss(nn.Module):
    """
    The Domain Adversarial Loss proposed in
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_
    [[Link to Source Code]](https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/alignment/dann.py)
    """
    def __init__(self, domain_discriminator, reduction='mean', grl=None):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = grl or WarmStartGradientReverseLayer(
            alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True
        ) 
        self.domain_discriminator = domain_discriminator
        self.reduction = reduction
        self.domain_discriminator_accuracy = None

    def forward(self, f_s, f_t, w_s=None, w_t=None):
        device = f_s.device
        # Concatenate source and target features, then apply gradient reversal
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        # Pass concatenated features through the domain discriminator
        d = self.domain_discriminator(f)
        
        batch_size_s, batch_size_t = f_s.size(0), f_t.size(0)
        
        # Split the discriminator output back into source/target
        d_s, d_t = torch.split(d, [batch_size_s, batch_size_t], dim=0)
        
        # Construct labels: source=1, target=0
        d_label_s = torch.ones(batch_size_s, 1, device=device)
        d_label_t = torch.zeros(batch_size_t, 1, device=device)
        
        # Calculate domain discriminator accuracy for monitoring
        with torch.no_grad():
            bin_accuracy = Accuracy(task='binary').to(device)
            acc_s = bin_accuracy(d_s, d_label_s).item()
            acc_t = bin_accuracy(d_t, d_label_t).item()
            self.domain_discriminator_accuracy = 0.5 * (acc_s + acc_t)

        # Set default weights if None
        w_s = torch.ones_like(d_label_s) if w_s is None else w_s.view_as(d_s)
        w_t = torch.ones_like(d_label_t) if w_t is None else w_t.view_as(d_t)
            
        # Compute BCE losses for source and target, then average
        loss_s = nn.functional.binary_cross_entropy(
            d_s, d_label_s, weight=w_s, reduction=self.reduction)
        loss_t = nn.functional.binary_cross_entropy(
            d_t, d_label_t, weight=w_t, reduction=self.reduction)
        return 0.5 * (loss_s + loss_t)
            
            
class ConditionalDomainAdversarialLoss(nn.Module):
    """The Conditional Domain Adversarial Loss used in 
    `Conditional Adversarial Domain Adaptation (NIPS 2018) <https://arxiv.org/abs/1705.10667>`_
    [[Link to Source Code]](https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/alignment/cdan.py)
    """
    def __init__(self, domain_discriminator, entropy_conditioning, grl=None, reduction='mean'):
        super(ConditionalDomainAdversarialLoss, self).__init__()
        self.grl = grl or WarmStartGradientReverseLayer(
            alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True
        )
        self.domain_discriminator = domain_discriminator
        self.entropy_conditioning = entropy_conditioning
        self.reduction = reduction
        self.domain_discriminator_accuracy = None
    
    def forward(self, f_s, l_s, f_t, l_t):
        device = f_s.device
        embeddings = torch.cat((f_s, f_t), dim=0)
        logits = torch.cat((l_s, l_t), dim=0)
        
        batch_size = embeddings.size(0)
        batch_size_s, batch_size_t = f_s.size(0), f_t.size(0)
        
        preds = torch.softmax(logits, dim=1).detach()
        log_preds = torch.log_softmax(logits, dim=1).detach()
        
        # Apply gradient reversal layer to the linear mapping of the embeddings and predictions
        h = self.grl(self._linear_map(embeddings, preds))
        d = self.domain_discriminator(h)
        
        d_label = torch.cat([
            torch.ones(batch_size_s, 1, device=device),
            torch.zeros(batch_size_t, 1, device=device)
        ])
        with torch.no_grad():
            bin_accuracy = Accuracy(task='binary').to(device)
            self.domain_discriminator_accuracy = bin_accuracy(d, d_label).item()
        
        if self.entropy_conditioning:
            # Entropy = - \sum p(x) log p(x)
            # Compute the entropy weight (1 + e^{-entropy}), then normalize by sum
            entropy = -(preds * log_preds).sum(dim=1)
            entropy_weight = 1.0 + torch.exp(-entropy)
            entropy_weight = entropy_weight / entropy_weight.sum() * batch_size
            weights = entropy_weight.view_as(d)
        else:
            weights = None
        return nn.functional.binary_cross_entropy(
            d, d_label, weight=weights, reduction=self.reduction
        )
        
    def _linear_map(self, f, y):
        batch_size = f.size(0)
        output = torch.bmm(y.unsqueeze(2), f.unsqueeze(1)) # shape: [batch_size, num_classes, feature_dim]
        return output.view(batch_size, -1) # shape: [batch_size, num_classes * feature_dim]
    
            
class MinimumClassConfusionLoss(nn.Module):
    """
    Minimum Class Confusion loss minimizes the class confusion in the target predictions.
    Proposed in `Minimum Class Confusion for Versatile Domain Adaptation (ECCV 2020) <https://arxiv.org/abs/1912.03699>`_
    [[Link to Source Code]](https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/self_training/mcc.py#L17)
    """
    def __init__(self, T):
        super(MinimumClassConfusionLoss, self).__init__()
        self.T = T

    def forward(self, logits):
        batch_size, num_classes = logits.shape
        
        # Temperature scaling
        preds = torch.softmax(logits / self.T, dim=1)         # shape: [batch_size, num_classes]
        log_preds = torch.log_softmax(logits / self.T, dim=1) # shape: [batch_size, num_classes]
        
        # Entropy = - \sum p(x) log p(x)
        with torch.no_grad():
            entropy = -(preds * log_preds).sum(dim=1) # shape: [batch_size]
            # Compute the entropy weight (1 + e^{-entropy}), then normalize by sum
            entropy_weight = 1.0 + torch.exp(-entropy)
            entropy_weight = (batch_size * entropy_weight / entropy_weight.sum()).unsqueeze(1)
            # shape of entropy_weight: [batch_size, 1]
        
        # Compute the class confusion matrix = P^T * P, weighed by the entropy weights
        # where P = softmax(logits)
        class_confusion_matrix = (preds * entropy_weight).transpose(0, 1).mm(preds)
        # Normalize each row
        class_confusion_matrix = class_confusion_matrix / class_confusion_matrix.sum(dim=1, keepdim=True)
        
        # MCC loss = (sum of matrix - trace of matrix) / num_classes
        # trace is the sum of diagonal elements
        mcc_loss = (class_confusion_matrix.sum() - class_confusion_matrix.trace()) / num_classes
        return mcc_loss     

     