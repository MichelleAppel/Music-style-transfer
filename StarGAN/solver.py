from model import Generator, Generator2, Generator3
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import gc


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, data_loader, config, data_size, c_dim):
        """Initialize configurations."""

        # Data loader.
        self.data_loader = data_loader

        # Model configurations.
        self.g_model = config.g_model
        self.data_size = data_size
        self.c_dim = c_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() and not config.force_cpu else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir
        self.data_dir = config.data_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.g_model == 2:
            self.G = Generator2(self.c_dim)
        elif self.g_model == 3:
            self.G = Generator3(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        else:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)

        self.D = Discriminator(self.data_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(
            outputs=y, inputs=x, grad_outputs=weight, retain_graph=True, create_graph=True, only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm - 1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, selected_attrs=None):
        """Generate target domain labels for debugging and testing."""

        c_trg_list = []
        for i in range(c_dim):
            c_trg = self.label2onehot(torch.ones(c_org.size(0)) * i, c_dim)
            c_trg_list.append(c_trg.to(self.device))

        return c_trg_list

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.cross_entropy(logit, target)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Fetch fixed inputs for debugging.
        data_iter = iter(self.data_loader)
        x_fixed, c_org = next(data_iter)
        result_shape = x_fixed.shape[-2:]
        np.save(self.sample_dir + '/original', (x_fixed[0].numpy().reshape(result_shape) + 1) / 2)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real spectograms and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            c_org = self.label2onehot(label_org, self.c_dim)
            c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device)  # Input.
            c_org = c_org.to(self.device)  # Original domain labels.
            c_trg = c_trg.to(self.device)  # Target domain labels.
            label_org = label_org.to(self.device)  # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)  # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real spectrograms.
            out_src, out_cls = self.D(x_real)
            d_loss_real = -torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org)

            # Compute loss with fake spectrograms.
            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)
            del x_fake
            del x_hat

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = -torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg)

                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                del g_loss_fake
                del g_loss_cls
                del x_reconst
                del g_loss_rec
                del g_loss

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i + 1)

            # Translate fixed spectrograms for debugging.
            if (i + 1) % self.sample_step == 0:
                sample_path = os.path.join(self.sample_dir, '{}-spectrograms'.format(i + 1))
                if not os.path.exists(sample_path):
                    os.mkdir(sample_path)

                with torch.no_grad():
                    x_fake_list = [x_fixed]

                    for j, c_fixed in enumerate(c_fixed_list):
                        generated = self.G(x_fixed, c_fixed)
                        x_fake_list.append(generated)
                        spectrogram = generated[0].cpu().numpy().reshape(result_shape)
                        np.save(sample_path + '/' + str(int(np.squeeze(np.array(c_fixed[0])).nonzero()[0])), (spectrogram + 1) / 2)

                    x_concat = torch.cat(x_fake_list, dim=3)
                    save_image(self.denorm(x_concat.data.cpu()), sample_path + '/visual.jpg', nrow=1, padding=0)

                    print('Saved real and fake spectrograms into {}...'.format(sample_path))

                del x_fake_list
                del x_concat

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i + 1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

            del x_real
            del label_org
            del rand_idx
            del label_trg
            del c_org
            del c_trg
            del out_src
            del out_cls
            del d_loss_real
            del d_loss_cls
            del alpha
            del d_loss_gp
            del d_loss
            del loss

    def test(self):
        """Translate spectrograms using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        # Set data loader.
        data_loader = self.data_loader

        with torch.no_grad():
            result_dir = self.result_dir

            dataset = str(self.data_dir)
            left = [i for i, ltr in enumerate(dataset) if ltr == '/'][-2]
            right = [i for i, ltr in enumerate(dataset) if ltr == '/'][-1]
            dataset = dataset[left + 1:right]

            destination = 'model_' + str(self.test_iters) + '_dataset_' + dataset
            if not os.path.exists(result_dir + '/' + destination):
                os.mkdir(result_dir + '/' + destination)

            for i, (x_real, c_org) in enumerate(data_loader):
                if not os.path.exists(result_dir + '/' + destination + '/' + str(i)):
                    os.mkdir(result_dir + '/' + destination + '/' + str(i))

                # Prepare input spectrograms and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.selected_attrs)

                x_fake_list = [x_real]

                # Reshape tensor to np array
                result_shape = x_real.shape[-2:]
                x_original = np.array(x_real).reshape(result_shape)

                # Save original file
                np.save(result_dir + '/' + destination + '/' + str(i) + '/original', x_original)

                # Save translations
                for c_trg in c_trg_list:
                    generated = self.G(x_real, c_trg)
                    x_fake_list.append(generated)
                    spectrogram = np.array(generated).reshape(result_shape)
                    np.save(
                        result_dir + '/' + destination + '/' + str(i) + '/' + str(
                            int(np.squeeze(np.array(c_trg)).nonzero()[0])), spectrogram)

                x_concat = torch.cat(x_fake_list, dim=3)
                save_image(
                    self.denorm(x_concat.data.cpu()),
                    result_dir + '/' + destination + '/' + str(i) + '/visual.jpg',
                    nrow=1,
                    padding=0)
                print('Saved real and fake spectrograms into {}...'.format(result_dir))

                del generated