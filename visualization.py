import cv2
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches
import numpy as np
import os
import re
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from add_ckpt_path import add_path_to_dust3r
add_path_to_dust3r("src/cut3r_512_dpt_4_64.pth")
from src.dust3r.utils.image import load_images


def load_feats(seq_dir, feat_type="state", feat_key="state_feat"):
    feats, skip_first = [], True
    for feat_fname in sorted(os.listdir(os.path.join(seq_dir, feat_type))):
        if feat_type == "state":
            if skip_first:    # Skip state 0, which is the initial state before reading images
                skip_first = False
                continue
        if (ext := os.path.splitext(feat_fname)[1]) == ".npz":
            with np.load(os.path.join(seq_dir, feat_type, feat_fname)) as feat_file:
                feat = feat_file[feat_key]
        else:
            feat = np.load(os.path.join(seq_dir, feat_type, feat_fname))
        feats.append(feat)
        print(f"load from {os.path.join(feat_type, feat_fname)}")
    if ext != ".npz" and feat_key is not None:
        print(f"{feat_type}'s file extension ({ext}) is not npz, ignoring feat_key ({feat_key})")
    return feats


def load_orig_imgs(seq_dir):
    natural_sort = lambda text: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]
    orig_images, img_paths = [], []
    for img_fname in sorted(os.listdir(seq_dir), key=natural_sort):
        if os.path.splitext(img_fname)[1] == ".png":
            img_path = os.path.join(seq_dir, img_fname)
            orig_images.append(plt.imread(img_path))
            print(f"Img read from {img_fname}")
            img_paths.append(img_path)
    return orig_images, img_paths


def visualize_tsne(fig, tsne_feats, orig_img, state_diff, xmin, ymin, xmax, ymax):
    axes = np.empty((3), dtype="object")
    token_colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(tsne_feats)))
    ratio = 10
    gspec = fig.add_gridspec(1, ratio * 2 if state_diff is None else ratio * 2 + 1, wspace=0.1)
    axes[0] = fig.add_subplot(gspec[0, :ratio])
    axes[1] = fig.add_subplot(gspec[0, ratio:ratio * 2] if state_diff is None else gspec[0, ratio + 1 : ratio * 2 + 1])
    if state_diff is not None:
        axes[2] = fig.add_subplot(gspec[0, ratio])
        axes[2].scatter(np.ones(len(state_diff)), np.arange(len(state_diff)), marker="_", color=plt.get_cmap("hot")(state_diff))
        axes[2].scatter(np.zeros(len(tsne_feats)), np.arange(len(tsne_feats)), marker="_", color=token_colors)
        axes[2].set_axis_off()
    axes[0].set_xlim(xmin, xmax)
    axes[0].set_ylim(ymin, ymax)
    # ax.set_zlim(min_coords[2], max_coords[2])
    axes[0].scatter(tsne_feats[:, 0], tsne_feats[:, 1], color=token_colors)   #, tsne_per_frame[frame_i][:, 2])
    axes[0].set_title("T-SNE")
    axes[1].imshow(orig_img)
    axes[1].set_title("Ground Truth View")
    axes[1].set_axis_off()
    return fig


def diff_between_state_tokens(state_feats):
    perct_diffs = []
    for frame_i in range(len(state_feats) - 1):
        """ 
        norm_prev, norm_next = np.linalg.norm(state_feats[frame_i], axis=1), np.linalg.norm(state_feats[frame_i + 1], axis=1)
        perct_diff = (norm_next - norm_prev) / norm_prev
        if np.any(perct_diff > 0):
            perct_diff[perct_diff > 0] /= np.max(perct_diff[perct_diff > 0])
        if np.any(perct_diff < 0):
            perct_diff[perct_diff < 0] /= - np.min(perct_diff[perct_diff < 0])
         """
        diff_norm = np.linalg.norm(state_feats[frame_i + 1] - state_feats[frame_i], axis=1)
        curr_norm = np.linalg.norm(state_feats[frame_i], axis=1)
        perct_diff = diff_norm / curr_norm
        perct_diff /= np.max(perct_diff)
        perct_diffs.append(perct_diff)
    return perct_diffs


def plot_tsne(seq_dir):
    state_feats = load_feats(seq_dir)
    orig_images, _ = load_orig_imgs(seq_dir)
    state_diffs = diff_between_state_tokens(state_feats)
    tsne_per_frame = []
    for frame_i in range(len(state_feats)):
        tsne_per_frame.append(TSNE(2).fit_transform(state_feats[frame_i]))
    max_coords = [max(np.max(frame_tsne[:, dim]) for frame_tsne in tsne_per_frame) for dim in range(2)]
    min_coords = [min(np.min(frame_tsne[:, dim]) for frame_tsne in tsne_per_frame) for dim in range(2)]

    tsne_save_dir = os.path.join(seq_dir, "visualizations", "tsne_vis")
    os.makedirs(tsne_save_dir, exist_ok=True)

    num_rows, num_cols = (len(orig_images) + 2) // 3, 3
    comb_fig = plt.figure(layout="constrained", figsize=(7 * num_cols, 4 * num_rows))
    sub_figs = comb_fig.subfigures(num_rows, num_cols)
    sub_figs = sub_figs.ravel()
    for frame_i in range(len(tsne_per_frame)):
        indiv_fig = plt.figure(layout="constrained", figsize=(15, 7))
        indiv_fig, sub_figs[frame_i] = [visualize_tsne(fig, tsne_per_frame[frame_i], orig_images[frame_i],
                                                       state_diffs[frame_i - 1] if frame_i > 0 else None, *min_coords, *max_coords)
                                        for fig in [indiv_fig, sub_figs[frame_i]]]
        indiv_fig.suptitle(f"CUT3R State Token Visualization for Input #{frame_i + 1}")
        indiv_fig.savefig(os.path.join(tsne_save_dir, f"input{frame_i + 1}.png"))
        sub_figs[frame_i].suptitle(f"Input #{frame_i + 1}")
    comb_fig.suptitle("CUT3R State Token Visualizations")
    comb_fig.savefig(os.path.join(tsne_save_dir, "combined.png"))


def generate_pca_plots(view_feats, orig_images, img_s, concat_pca=True, outlier_thresh=0):
    def calc_pca_with_outliers(features, out_std):
        pca = PCA(n_components=3)
        pca_feats = pca.fit_transform(features)
        outliers = []
        for i in range(3):
            # pca_feats[:, i] = (pca_feats[:, i] - pca_feats[:, i].min()) / (pca_feats[:, i].max() - pca_feats[:, i].min())
            # transform using mean and std
            pca_mean, pca_std = pca_feats[:, i].mean(), pca_feats[:, i].std()
            if out_std > 0:
                out_dim = np.logical_or(pca_feats[:, i] < pca_mean - out_std * pca_std, pca_feats[:, i] > pca_mean + out_std * pca_std)
                outliers.append(out_dim)
            pca_feats[:, i] = (pca_feats[:, i] - pca_mean) / (pca_std * out_std * 2 if out_std > 0 else pca_std ** 2) + 0.5
        
        if out_std > 0:
            final_out = np.vstack(outliers).any(axis=0)
            print(f"PCA total outliers: {np.sum(final_out)} (# samples: {len(pca_feats)})")
            out_fig, out_axes = plt.subplots(3, 1, figsize=(10, 8), layout="constrained")
            for i in range(3):
                out_axes[i].scatter(pca_feats[:, i][outliers[i]], np.ones(np.sum(outliers[i])), color="red")
                out_axes[i].scatter(pca_feats[:, i][~outliers[i]], np.ones(np.sum(~outliers[i])), color="green")
                out_axes[i].set_title(f"PCA Values (Dimension {i})")
                out_axes[i].tick_params(axis="y", left=False, labelleft=False)
            # Manual in/outliers legend with counts
            out_handle = matplotlib.patches.Patch(color="red", label=f"Outliers ({np.sum(final_out)})")
            in_handle = matplotlib.patches.Patch(color="green", label=f"Inliers ({len(final_out) - np.sum(final_out)})")
            out_fig.legend(handles=[out_handle, in_handle], loc="outside lower center")
            out_fig.suptitle("PCA Inliers / Outliers")
            plt.close()
            pca_feats[final_out] = 0
            return pca_feats, out_fig
        return pca_feats
    
    if concat_pca:
        agg_feats = np.concatenate(view_feats, axis=0)
        pca_feats = calc_pca_with_outliers(agg_feats, outlier_thresh)
        if outlier_thresh > 0:
            pca_feats, out_fig = pca_feats
    else:
        pca_feats = []
        for state_feat in view_feats:
            pass
        


def plot_pca(state_feats, orig_images, fig_save_dir, img_s, head_i, concat_pca=True):
    if concat_pca:
        agg_feats = np.concatenate(state_feats, axis=0)
        pca = PCA(n_components=3)
        pca_feats = pca.fit_transform(agg_feats)
        outliers = []
        for i in range(3):
            # pca_feats[:, i] = (pca_feats[:, i] - pca_feats[:, i].min()) / (pca_feats[:, i].max() - pca_feats[:, i].min())
            # transform using mean and std
            pca_mean, pca_std = pca_feats[:, i].mean(), pca_feats[:, i].std()
            out_dim = np.logical_or(pca_feats[:, i] < pca_mean - 2 * pca_std, pca_feats[:, i] > pca_mean + 2 * pca_std)
            outliers.append(out_dim)
            pca_feats[:, i] = (pca_feats[:, i] - pca_mean) / (pca_std * 4) + 0.5
        
        final_out = np.vstack(outliers).any(axis=0)
        print(f"Total outliers: {np.sum(final_out)} (total: {len(pca_feats)})")
        
        out_fig, out_axes = plt.subplots(3, 1, figsize=(10, 8), layout="constrained")
        for i in range(3):
            out_axes[i].scatter(pca_feats[:, i][outliers[i]], np.ones(np.sum(outliers[i])), color="red")
            out_axes[i].scatter(pca_feats[:, i][~outliers[i]], np.ones(np.sum(~outliers[i])), color="green")
            out_axes[i].set_title(f"PCA Values (Dimension {i})")
            out_axes[i].tick_params(axis="y", left=False, labelleft=False)
        # Manual in/outliers legend with counts
        out_handle = matplotlib.patches.Patch(color="red", label=f"Outliers ({np.sum(final_out)})")
        in_handle = matplotlib.patches.Patch(color="green", label=f"Inliers ({len(final_out) - np.sum(final_out)})")
        out_fig.legend(handles=[out_handle, in_handle], loc="outside lower center")
        out_fig.suptitle("PCA Inliers / Outliers")
        out_fig.savefig(os.path.join(fig_save_dir, f"head{head_i}_outliers.png"))
        plt.close()
        pca_feats[final_out] = 0
        
    else:
        pca_feats = []
        for state_feat in state_feats:
            pca_feat = PCA(n_components=3).fit_transform(state_feat)
            for i in range(3):
                # pca_feat[:, i] = (pca_feat[:, i] - pca_feat[:, i].min()) / (pca_feat[:, i].max() - pca_feat[:, i].min())
                pca_feat[:, i] = (pca_feat[:, i] - pca_feat[:, i].mean()) / (pca_feat[:, i].std() * 2) + 0.5
            pca_feats.append(pca_feat)
    
    num_rows, num_cols = (len(orig_images) + 2) // 3, 3
    comb_fig = plt.figure(layout="constrained", figsize=(6 * num_cols, 2.5 * num_rows))
    sub_figs = comb_fig.subfigures(num_rows, num_cols)
    sub_figs = sub_figs.ravel()

    for img_i in range(len(orig_images)):
        pca_per_img = pca_feats[img_i * (img_s[0] // PATCH) * (img_s[1] // PATCH) : (img_i + 1) * (img_s[0] // PATCH) * (img_s[1] // PATCH)] if concat_pca else \
                        pca_feats[img_i]
        axes = sub_figs[img_i].subplots(1, 2)
        feat_map = pca_per_img.reshape(img_s[1] // PATCH, img_s[0] // PATCH, -1)
        # feat_map = cv2.resize(feat_map, (img_s[0], img_s[1]))
        axes[0].imshow(orig_images[img_i])
        axes[1].imshow(feat_map[..., ::-1])
    comb_fig.suptitle("PCA of State Tokens Weighted Sum According to Cross Attentions")
    comb_fig.savefig(os.path.join(fig_save_dir, f"head{head_i}.png"))
    plt.close()


def cross_attns_pca(seq_dir, state_to_img_type, img_s):
    vis_dir = os.path.join(seq_dir, "visualizations", state_to_img_type)
    os.makedirs(vis_dir, exist_ok=True)
    attn_type = "image"
    cross_attns = load_feats(seq_dir, "cross_attns", attn_type)
    state_feats = load_feats(seq_dir)
    orig_images, _ = load_orig_imgs(seq_dir)
    os.makedirs(os.path.join(vis_dir, f"{attn_type}_cross_attn"), exist_ok=True)
    layer_i = 0
    for head_i in range(cross_attns[0].shape[1]):
        states_remap = []
        for view_i in range(len(orig_images)):
            attn_map = cross_attns[view_i][layer_i, head_i]
            states_weights = attn_map[1:] if attn_type == "image" else attn_map[:, 1:]
            if state_to_img_type == "cross_attns_weighte_sum":
                state_remap = (state_feats[view_i].T @ states_weights).T
            elif state_to_img_type == "cross_attns_top_1":
                state_remap = state_feats[view_i][np.argmax(states_weights, axis=1)]
            states_remap.append(state_remap)
        plot_pca(states_remap, orig_images, os.path.join(vis_dir, f"{attn_type}_cross_attn"), img_s, head_i)


def cross_attns_map(seq_dir, img_s, token_i=30):
    vis_dir = os.path.join(seq_dir, "visualizations", f"cross_attn_map_state_token_{token_i}")
    os.makedirs(vis_dir, exist_ok=True)
    cross_attns = load_feats(seq_dir, "cross_attns", "state")
    orig_images, _ = load_orig_imgs(seq_dir)
    layer_i = 0

    num_rows, num_cols = (len(orig_images) + 2) // 3, 3
    for head_i in range(cross_attns[0].shape[1]):
        head_fig = plt.figure(layout="constrained", figsize=(6 * num_cols, 2.5 * num_rows)) 
        sub_figs = head_fig.subfigures(num_rows, num_cols)
        sub_figs = sub_figs.ravel()

        for view_i in range(len(orig_images)):
            axes = sub_figs[view_i].subplots(1, 2)
            state_attns = cross_attns[view_i][layer_i, head_i, token_i]
            attn_map = state_attns[1:].reshape((img_s[1] // PATCH, img_s[0] // PATCH))
            attn_scaled = cv2.resize(attn_map, orig_images[view_i].shape[1::-1])            
            axes[0].imshow(orig_images[view_i])
            axes[1].imshow(orig_images[view_i])
            axes[1].imshow(attn_scaled, cmap="inferno", alpha=0.6)
        head_fig.savefig(os.path.join(vis_dir, f"head{head_i}.png"))
        plt.close()


def state_token_change_cumulative_ranking(seq_dir):
    state_feats = load_feats(seq_dir)
    state_diffs = diff_between_state_tokens(state_feats)
    token_score = np.zeros(len(state_diffs[0]))
    for perct_diff in state_diffs:
        token_score += np.arange(len(perct_diff))[np.argsort(perct_diff)]
    token_rank = np.argsort(token_score)
    token_rank_str = [str(token_i) for token_i in token_rank]
    rank_fig, rank_ax = plt.subplots(figsize=(100, 10))
    rank_ax.bar(token_rank_str, token_score[token_rank])
    rank_ax.tick_params(axis="x", labelrotation=90, labelsize=10)
    rank_fig.savefig(os.path.join(seq_dir, "visualizations", "cumulative_token_changes.png"))
    plt.close()
    with open(os.path.join(seq_dir, "visualizations", "state_token_change_ranking.txt"), "w") as rank_file:
        rank_file.write(" ".join(token_rank_str[::-1]))


if __name__ == "__main__":
    IMG_S, PATCH = 512, 16
    seq_root = "/mnt/NAS/data/gj2148/CUT3R/"
    
    seq_dirs = ["test_sequences/cont_seq_cut3r_restart/cut3r_2nd_half",
                "test_sequences/cont_seq_cut3r_restart/cut3r_1st_half",
                "test_sequences/objcentric_disjoint_2views/same_cut3r_instance",
                "test_sequences/objcentric_disjoint_2views_2/diff_cut3r_instance_view2",
                "test_sequences/objcentric_disjoint_2views_2/diff_cut3r_instance_view1",
                "test_sequences/percep_aliasing_2views/loc1_new_cut3r",
                "test_sequences/objcentric_disjoint_2views_2/same_cut3r_instance",
                "test_sequences/objcentric_disjoint_2views/diff_cut3r_instance_view1",
                "test_sequences/objcentric_disjoint_2views/diff_cut3r_instance_view2",
                "test_sequences/percep_aliasing_2views/loc2_new_cut3r",
                "test_sequences/opposite_views",
                "test_sequences/no_visual_overlap",
                "test_sequences/cont_seq_cut3r_restart/cut3r_full_seq"]
    for seq_dir in seq_dirs[-1:]:
        # plot_tsne(seq_dir)
        _, img_paths = load_orig_imgs(os.path.join(seq_root, seq_dir))
        img_shape = load_images(img_paths[:1], IMG_S)[0]["true_shape"][0, ::-1]    # 1st list item; remove batch dim in img shape
        
        state_to_img_type = "cross_attns_top_1"    # "cross_attns_top_1"
        cross_attns_pca(os.path.join(seq_root, seq_dir), state_to_img_type, img_shape)
        
        """ 
        state_token_change_cumulative_ranking(seq_dir)
        with open(os.path.join(seq_dir, "visualizations", "state_token_change_ranking.txt"), "r") as rank_file:
            token_ranks = rank_file.read().split()
        for token_i in token_ranks[:5] + token_ranks[-5:]:
            cross_attns_map(seq_dir, img_shape, int(token_i))
        with open(os.path.join(seq_dir, "visualizations", "state_token_change_top5.txt"), "w") as top5_file:
            top5_file.write(f"Top 5 changed tokens: {' '.join(token_ranks[:5])}\nBottom 5 changed tokens: {' '.join(token_ranks[-5:])}")
         """


""" Cosine similarity is of course not possible since image tokens and state tokens are different length (1024 vs 768)
if __name__ == "__main__":
    seq_dir = "test_sequences/objcentric_disjoint_2views/same_cut3r_instance"
    state_feats = load_feats(seq_dir)
    img_feats = load_feats(seq_dir, "img_tokens", None)
    for view_i in range(len(state_feats)):
        cos_sims = cosine_similarity(state_feats[view_i], img_feats[view_i])
        top_img_per_state = np.argmax(cos_sims, axis=1)
        print(f"{view_i}:", len(np.unique(top_img_per_state)) == len(top_img_per_state))
"""