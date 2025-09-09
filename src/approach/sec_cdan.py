import sys
import torch
from torch import nn
from tqdm import tqdm

from approach.dl_module import DLModule
from data.util import InfiniteDataIterator
from module.domain_discriminator import DomainDiscriminator
from module.gradient_reverse_function import WarmStartGradientReverseLayer
from module.loss import ConditionalDomainAdversarialLoss
from util.config import load_config

disable_tqdm = not sys.stdout.isatty()


class SecCDAN(DLModule):
    """
    [[Link to Source Code]](https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/cdan.py)
    CDAN is a class that implements the approach described in "Conditional Adversarial Domain Adaptation".
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()
        
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.discr_hidden_size = kwargs.get('discr_hidden_size', cf['discr_hidden_size'])
        self.entropy = kwargs.get('cdan_entropy', cf['cdan_entropy'])
        self.cdan_alpha = kwargs.get('cdan_alpha', cf['cdan_alpha'])
        self.num_classes = kwargs.get('num_classes', cf['num_classes'])
        self.adapt_lr = kwargs.get('adapt_lr', cf['adapt_lr'])
        self.adapt_epochs = kwargs.get('adapt_epochs', cf['adapt_epochs'])
        self.iter_per_epoch = kwargs.get('iter_per_epoch', cf['iter_per_epoch'])
        
        # Domain adaptation setup
        self.domain_discriminator = DomainDiscriminator(
            in_feature=self.net.out_features_size * kwargs['num_classes'],
            hidden_size=self.discr_hidden_size,
        )
        wsgrl = WarmStartGradientReverseLayer(**kwargs)
        self.da_loss = ConditionalDomainAdversarialLoss(
            self.domain_discriminator, 
            entropy_conditioning=self.entropy, 
            grl=wsgrl,
        ).to(self.device)
        
    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = DLModule.add_appr_specific_args(parent_parser)
        parser.add_argument('--discr-hidden-size', type=int, default=cf['discr_hidden_size'])
        parser.add_argument('--cdan-entropy', action='store_true', default=cf['cdan_entropy'])
        parser.add_argument('--cdan-alpha', type=float, default=cf['cdan_alpha'])
        parser.add_argument('--adapt-lr', type=float, default=cf['adapt_lr'])
        parser.add_argument('--adapt-epochs', type=int, default=cf['adapt_epochs'])
        parser.add_argument('--iter-per-epoch', type=int, default=cf['iter_per_epoch'])
        return parser
    
    
    def _fit_step(self, batch_x, batch_y):
        logits = self.net(batch_x)
        loss = self.ce_loss(logits, batch_y)
        return loss, logits
    
    
    def _predict_step(self, batch_x, batch_y):
        logits = self.net(batch_x)
        loss = self.ce_loss(logits, batch_y)
        return loss, logits
    
    
    def _adapt(self, adapt_dataloader, val_dataloader, train_dataloader):
        # Infinite iterators
        src_dataloader = InfiniteDataIterator(train_dataloader, device=self.device) 
        trg_dataloader = InfiniteDataIterator(adapt_dataloader, device=self.device) 

        self.lr = self.adapt_lr
        params = list(self.net.parameters()) + list(self.domain_discriminator.parameters())
        self.configure_optimizers(params=params)
        
        # Adaptation loop       
        for epoch in range(self.adapt_epochs):
            self.net.train()
            self.da_loss.train()
            
            for cb in self.callbacks:
                cb.on_epoch_start(self, epoch)
                
            running_loss = 0.0
            
            postfix = {
                'DA loss': f'{self.epoch_outputs["train_loss"]:.4f}', 
                'discr acc':  f'{self.epoch_outputs["train_accuracy"]:.4f}',
                f'val {self.sch_monitor}': f'{val_score:.4f}'
            } if epoch > 0 else {} 
            adapt_loop = tqdm(
                range(self.iter_per_epoch), desc=f'Ep[{epoch+1}/{self.adapt_epochs}]',  
                postfix=postfix, leave=False, disable=disable_tqdm
            )
            for i in adapt_loop:
                # Get batches
                batch_x_s, batch_y_s = next(src_dataloader)
                batch_x_t, _ = next(trg_dataloader)
                batch_x_s, batch_y_s = batch_x_s.to(self.device), batch_y_s.to(self.device).long()
                batch_x_t = batch_x_t.to(self.device)
                
                # Compute logits and embeddings
                batch_size_s, batch_size_t = batch_x_s.size(0), batch_x_t.size(0)
                batch_x = torch.cat((batch_x_s, batch_x_t), dim=0)
                logits, batch_emb = self.net(batch_x, return_feat=True)
                logits_s, logits_t = torch.split(logits, [batch_size_s, batch_size_t], dim=0)
                batch_emb_s, batch_emb_t = torch.split(batch_emb, [batch_size_s, batch_size_t], dim=0)
                
                # Compute losses
                ce_loss = self.ce_loss(logits_s, batch_y_s)
                da_loss = self.da_loss(f_s=batch_emb_s, l_s=logits_s, f_t=batch_emb_t, l_t=logits_t)
                loss = ce_loss + da_loss * self.cdan_alpha
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                
            # Validation on adapt epoch end
            self.epoch_outputs = self._predict(val_dataloader, on_train_epoch_end=True)
            val_score = self.epoch_outputs[self.sch_monitor]
            
            self.epoch_outputs['train_loss'] = running_loss / self.iter_per_epoch
            self.epoch_outputs['train_accuracy'] = self.da_loss.domain_discriminator_accuracy
            
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch)
                
            if self.should_stop:
                break  # Early stopping
            
            self.run_scheduler_step(monitor_value=val_score, epoch=epoch + 1)
        