import argparse
import re
import matplotlib.pyplot as plt
import json


def main(file_path):
    with open(file_path) as f:
        line = f.readline()
        line = line[10:-2]
        line = line.replace("'", '"')
        line = line.replace('=', '":')
        line = line.replace(', ', ', "')
        line = line.replace('False', 'false')
        line = line.replace('None', 'null')
        hyperparams = json.loads('{"' + line + '}')
        while line != 'Start training...\n':
            line = f.readline()

        iterations = []
        times = []
        D_losses_real = []
        D_losses_fake = []
        D_losses_cls = []
        D_losses_gp = []
        D_losses = []
        G_losses_fake = []
        G_losses_rec = []
        G_losses_cls = []
        G_losses = []
        prog = re.compile(
            'Elapsed \[(.+)\], Iteration \[(.+)\/.+\], D\/loss_real: (.+), D\/loss_fake: (.+), D\/loss_cls: (.+), D\/loss_gp: (.+), G\/loss_fake: (.+), G\/loss_rec: (.+), G\/loss_cls: (.+)'
        )
        for line in f:
            try:
                time, iteration, D_loss_real, D_loss_fake, D_loss_cls, D_loss_gp, G_loss_fake, G_loss_rec, G_loss_cls = re.search(
                    prog, line).groups()
                times.append(time)
                iterations.append(int(iteration))
                D_losses_real.append(float(D_loss_real))
                D_losses_fake.append(float(D_loss_fake))
                D_losses_cls.append(float(D_loss_cls))
                D_losses_gp.append(float(D_loss_gp))
                D_losses.append(
                    float(D_loss_real) + float(D_loss_fake) + hyperparams['lambda_cls'] * float(D_loss_cls) +
                    hyperparams['lambda_gp'] * float(D_loss_gp))
                G_losses_fake.append(float(G_loss_fake))
                G_losses_rec.append(float(G_loss_rec))
                G_losses_cls.append(float(G_loss_cls))
                G_losses.append(
                    float(G_loss_fake) + hyperparams['lambda_rec'] * float(G_loss_rec) +
                    hyperparams['lambda_cls'] * float(G_loss_cls))
            except AttributeError:
                print(line)
                pass
        plt.plot(iterations, G_losses, iterations, D_losses)
        plt.show()
    return


def parse_line(line):

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default='log.txt', help='set path of log file')
    args = parser.parse_args()
    main(args.f)