import argparse
import re
import matplotlib.pyplot as plt


def main(file_path):
    with open(file_path) as f:
        line = f.readline()
        while line != 'Start training...\n':
            line = f.readline()

        iterations = []
        times = []
        D_losses_real = []
        D_losses_fake = []
        D_losses_cls = []
        G_losses_fake = []
        G_losses_rec = []
        G_losses_cls = []

        prog = re.compile(
            'Elapsed \[(.+)\], Iteration \[(.+)\/.+\], D\/loss_real: (.+), D\/loss_fake: (.+), D\/loss_cls: (.+), D\/loss_gp: (.+), G\/loss_fake: (.+), G\/loss_rec: (.+), G\/loss_cls: (.+)'
        )
        for line in f:
            try:
                time, iteration, D_loss_real, D_loss_fake, D_loss_cls, G_loss_gp, G_loss_fake, G_loss_rec, G_loss_cls = re.search(
                    prog, line).groups()
                times.append(time)
                iterations.append(int(iteration))
                D_losses_real.append(float(D_loss_real))
                D_losses_fake.append(D_loss_fake)
                D_losses_cls.append(D_loss_cls)
                G_losses_fake.append(G_loss_fake)
                G_losses_rec.append(G_loss_rec)
                G_losses_cls.append(G_loss_cls)
            except:
                print(line)
        plt.plot(iterations, D_losses_real)
        plt.show()
    return


def parse_line(line):

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default='log.txt', help='set path of log file')
    args = parser.parse_args()
    main(args.f)