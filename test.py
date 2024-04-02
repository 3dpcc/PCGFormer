import torch
import numpy as np
import os
from pcc_model import PCCModel
from pcc_model_pos import PCCModel_pos
from coder import Coder
from coder_pos import Coder_pos
import time
from data_utils import load_sparse_tensor, sort_sparse_tensor, scale_sparse_tensor
from data_utils import write_ply_ascii_geo
from pc_error import pc_error
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(filedir, ckptdir, outdir, resultdir, scaling_factor=1.0, rho=1.0, res=1024):
    # load data
    start_time = time.time()
    x = load_sparse_tensor(filedir, device)
    print('Loading Time:\t', round(time.time() - start_time, 4), 's')
    # x = sort_spare_tensor(input_data)

    # output filename
    if not os.path.exists(outdir): os.makedirs(outdir)
    filename = os.path.join(outdir, os.path.split(filedir)[-1].split('.')[0])
    print('output filename:\t', filename)
    # load model
    if (args.pct_pos == False):
        model = PCCModel().to(device)
    if (args.pct_pos == True ):
        model = PCCModel_pos().to(device)

    # print('parameter:')
    # print(sum(param.numel() for param in model.parameters()))

    assert os.path.exists(ckptdir)
    ckpt = torch.load(ckptdir,map_location='cuda:0')
    model.load_state_dict(ckpt['model'])
    print('load checkpoint from \t', ckptdir)
    # postfix: rate index
    postfix_idx = '_r' + str(idx + 1)
    if ( args.pct_pos == False):
        coder = Coder(model=model, filename=filename)
    if ( args.pct_pos == True):
        coder = Coder_pos(model=model, filename=filename)

    # down-scale
    if scaling_factor!=1:
        x_in = scale_sparse_tensor(x, factor=scaling_factor)
    else:
        x_in = x

    # encode
    start_time = time.time()
    _ = coder.encode(x_in, postfix=postfix_idx)
    print('Enc Time:\t', round(time.time() - start_time, 3), 's')
    time_enc = round(time.time() - start_time, 3)

    # decode
    start_time = time.time()
    x_dec = coder.decode(postfix=postfix_idx, rho=rho)
    print('Dec Time:\t', round(time.time() - start_time, 3), 's')
    time_dec = round(time.time() - start_time, 3)

    # up-scale
    if scaling_factor!=1:
        x_dec = scale_sparse_tensor(x_dec, factor=1.0/scaling_factor)

    # bitrate
    bits = np.array([os.path.getsize(filename + postfix_idx + postfix) * 8 \
                         for postfix in['_C.bin', '_F.bin', '_H.bin', '_num_points.bin']])

    # rate
    bpps = (bits/len(x)).round(3)
    print('bits:\t', sum(bits), '\nbpps:\t',  sum(bpps).round(3))

    # distortion
    start_time = time.time()
    write_ply_ascii_geo(filename+postfix_idx+'_dec.ply', x_dec.C.detach().cpu().numpy()[:,1:])
    print('Write PC Time:\t', round(time.time() - start_time, 3), 's')

    start_time = time.time()
    pc_error_metrics = pc_error(filedir, filename+postfix_idx+'_dec.ply',res=res, normal=True, show=False)

    print('PC Error Metric Time:\t', round(time.time() - start_time, 3), 's')
    print('D1 PSNR:\t', pc_error_metrics["mseF,PSNR (p2point)"][0])
    print('D2 PSNR:\t', pc_error_metrics["mseF,PSNR (p2plane)"][0])

    ######all_results#######
    results = pc_error_metrics
    results["num_points(input)"] = len(x)
    results["num_points(output)"] = len(x_dec)
    results["resolution"] = res
    results["bits"] = sum(bits).round(3)
    results["bits"] = sum(bits).round(3)
    results["bpp"] = sum(bpps).round(3)
    results["bpp(coords)"] = bpps[0]
    results["bpp(feats)"] = bpps[1]
    results["time(enc)"] = time_enc
    results["time(dec)"] = time_dec

    return results

if __name__ == '__main__':
    import argparse,glob
    from tqdm import tqdm
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filedir", default='/media/ivc-18/958e2f20-9a21-425d-8061-48542c9ca6c5/testdata/solid_all/10bit/')
    parser.add_argument("--outdir", default='./output')
    parser.add_argument("--resultdir", default='./results')
    parser.add_argument("--scaling_factor", type=float, default=1.0, help='scaling_factor')
    parser.add_argument("--res", type=int, default=1023, help='resolution')
    parser.add_argument("--ckptdir", default='./ckpt_pretrain/r02_0.25bpp/epoch_last.pth')
    # parser.add_argument("--hyper", action='store_true', help=" hpyer coder.")
    parser.add_argument("--pct_pos", action='store_true', help="position encoding for knn_transformer (R01).")
    parser.add_argument("--rho", type=float, default=1.0, help='the ratio of the number of output points to the number of input points')
    args = parser.parse_args()

    if not os.path.exists(args.outdir): os.makedirs(args.outdir)
    if not os.path.exists(args.resultdir): os.makedirs(args.resultdir)


    filedir_list = sorted(glob.glob(os.path.join(args.filedir,f'*.*'), recursive=True))
    list_orifile = [f for f in filedir_list if
                   f.endswith('ply')]
    idx = 0
    for file_name in tqdm(list_orifile):
            print("Coding PC:", file_name)
            results = test(file_name, args.ckptdir, args.outdir, args.resultdir, scaling_factor=args.scaling_factor, rho=args.rho, res=args.res)
            if idx == 0:
                all_results = results.copy(deep=True)
            else:
                all_results = all_results.append(results, ignore_index=True)
            csv_name = os.path.join(args.resultdir,str(args.res)+ '_RD_results.csv')
            all_results.to_csv(csv_name, index=False)
            print('Wrile results to: \t', csv_name)
            idx += 1


