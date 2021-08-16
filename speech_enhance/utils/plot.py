import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

import numpy as np

from .logger import log


def plot_alignment(alignment, path):
    alignment = np.where(alignment < 1, alignment, 1)
    # log("min and max:", np.min(alignment), np.max(alignment))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close()
    return


def plot_spectrogram(pred_spectrogram, plot_path, title="mel-spec", show=False):
    fig = plt.figure(figsize=(20, 10))
    fig.text(0.5, 0.18, title, horizontalalignment='center', fontsize=16)
    vmin = np.min(pred_spectrogram)
    vmax = np.max(pred_spectrogram)
    ax2 = fig.add_subplot(111)
    im = ax2.imshow(np.rot90(pred_spectrogram), interpolation='none',
                    vmin=vmin, vmax=vmax)
    char = fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal',
                        ax=ax2)

    # char.set_ticks(np.arange(vmin, vmax))
    char.set_ticks(np.arange(0, 1))

    plt.tight_layout()
    plt.savefig(plot_path, format='png')
    if show:
        plt.show()
    plt.close()
    log("save spec png to {}".format(plot_path))
    return


def plot_two_spec(pred_spec, target_spec, pic_path, title=None,
                  vmin=None, vmax=None):
    # assert np.shape(pred_spec)[1] == 80 and np.shape(target_spec)[1] == 80
    fig = plt.figure(figsize=(12, 8))
    fig.text(0.5, 0.18, title, horizontalalignment='center', fontsize=16)
    if vmin is None or vmax is None:
        vmin = min(np.min(pred_spec), np.min(target_spec))
        vmax = max(np.max(pred_spec), np.max(target_spec))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Predicted Mel-Spectrogram')
    im = ax1.imshow(np.rot90(pred_spec), interpolation='none',
                    vmin=vmin, vmax=vmax)
    fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax1)

    ax2 = fig.add_subplot(212)
    ax2.set_title('Target Mel-Spectrogram')
    im = ax2.imshow(np.rot90(target_spec), interpolation='none',
                    vmin=vmin, vmax=vmax)
    fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax2)

    plt.tight_layout()
    plt.savefig(pic_path, format='png')
    plt.close()
    log("save spec png to {}".format(pic_path))
    return


def plot_line(path, x_list, y_list, label_list):
    assert len(x_list) == len(y_list) == len(label_list)
    plt.title('Result Analysis')
    for x_data, y_data, label in zip(x_list, y_list, label_list):
        plt.plot(x_data, y_data, label=label)
        # plt.plot(x2, y2, color='red', label='predict')
    plt.legend()  # 显示图例
    plt.xlabel('frame-index')
    plt.ylabel('value')
    plt.savefig(path, format='png')
    plt.close()
    return


def plot_line_phone_time(path, time_pitch_index, pitch_seq, seq_label):
    """ show phoneme and pitch in time"""
    seq_str = []
    seq_index = []
    pre = seq_label[0]
    counter = 0
    for i in seq_label:
        if i != pre:
            seq_str.append(pre)
            seq_index.append(counter)
            pre = i
        else:
            counter += 1
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(time_pitch_index, pitch_seq, 'r')
    ax1.set_ylabel('pitch')
    for i in range(len(seq_index)):
        plt.vlines(seq_index[i], ymin=0, ymax=700)
        plt.text(seq_index[i] - 1, 800, seq_str[i], rotation=39)

    plt.savefig(path, format='png')
    return


def plot_mel(mel, path, info=None):
    mel = mel.T
    fig, ax = plt.subplots()
    im = ax.imshow(
        mel,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.show()
    plt.savefig(path, format='png')
    plt.close()

    return fig


def plot_one_mel_pitch_energy(mel, pitch, energy, stat_json_file, title, path):
    """plot mel/pitch/energy

    Args:
        mel: [dim, T]
        pitch: [T]
        energy: [T]
        stat_json_file: stat file
        title: titile
        path: path for png

    """
    with open(stat_json_file) as f:
        stats = json.load(f)
        stats = stats["phn_pitch"] + stats["phn_energy"][:2]
    # stats = [min(pitch), max(pitch), 0, 1.0, min(energy), max(energy)]

    fig = plot_multi_mel_pitch_energy(
        [
            (mel, pitch, energy),
        ],
        stats,
        [title],
    )
    plt.savefig(path, format='png')
    plt.close()


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def plot_multi_mel_pitch_energy(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False, figsize=(6.4, 3.0 * len(data)))
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        # log(titles, mel.shape, pitch.shape, energy.shape)
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        # log(pitch)
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, max(mel.shape[1], len(pitch)))
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, max(mel.shape[1], len(energy)))
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


if __name__ == '__main__':
    pass
