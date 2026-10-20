"""Original infographic and staged silent animation from executed results."""
import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'
INK, MUTED, BLUE, PURPLE, ORANGE, TEAL = '#15263C','#46566A','#1762A5','#7048A5','#A34D14','#146F70'


def poster(data, stage=4):
    fig = plt.figure(figsize=(10.8,13.5),dpi=100,facecolor='#F6F9FC')
    ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,1080);ax.set_ylim(1350,0);ax.axis('off')
    def text(x,y,s,size=25,color=INK,weight='normal',ha='left'):
        return ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va='top',fontfamily='DejaVu Sans',linespacing=1.35)
    def box(x,y,w,h,color='#FFFFFF',edge='#CFD9E5'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=16',facecolor=color,edgecolor=edge,linewidth=1.5))
    text(54,40,'Inside one LSTM step',34,weight='bold')
    text(54,104,'Keep memory. Write an update. Expose an output.',21)
    text(54,151,'CONSTRUCTED SCALAR EXAMPLE · NO TRAINING',15,MUTED,weight='bold')
    box(45,200,990,115)
    text(72,221,'Input x = 1.0     Previous h = 0.2',23)
    text(72,265,'Previous cell memory c = 1.50',23,weight='bold')
    if stage>=1:
        text(54,339,'1   Affine transforms → gate values',25,weight='bold')
        names=[('Forget  f','0.80',BLUE),('Input  i','0.25',PURPLE),('Candidate  g','−0.40',ORANGE),('Output  o','0.60',TEAL)]
        for index,(label,value,color) in enumerate(names):
            x=45+index*251;box(x,395,237,112,edge=color)
            text(x+118,412,label,19,color,weight='bold',ha='center')
            text(x+118,450,value,29,color,weight='bold',ha='center')
        text(54,526,'i, f, o = sigmoid(logit)       g = tanh(logit)',21,MUTED)
    if stage>=2:
        box(45,589,990,216,'#EAF3FC',BLUE)
        text(71,611,'2   Update the cell memory',25,BLUE,weight='bold')
        text(71,661,'c next = f × c previous + i × g',25,weight='bold')
        text(71,709,'0.80 × 1.50   +   0.25 × (−0.40)',25)
        text(71,752,'1.20 retained  −  0.10 written  =  1.10',25,BLUE,weight='bold')
    if stage>=3:
        box(45,831,990,154,'#E9F5F3',TEAL)
        text(71,850,'3   Expose the hidden output',25,TEAL,weight='bold')
        text(71,904,'h next = 0.60 × tanh(1.10) ≈ 0.4803',26,weight='bold')
        text(71,951,'The output gate scales tanh(c); it does not replace c.',17,MUTED)
    if stage>=4:
        box(45,1011,990,229)
        text(71,1031,'4   Keep the tensors straight',25,weight='bold')
        text(71,1082,'One step: x (B, D)     h and c (B, H)',23)
        text(71,1130,'Batch-first sequence: (B, T, D)',23)
        text(71,1178,'Cell memory c and hidden output h are different.',21,BLUE,weight='bold')
    text(54,1262,'Standard cell; no projection or peepholes. PyTorch 2.9 equations.',14,MUTED)
    text(54,1295,'Schematic stages; the 20-second playback is not execution time.',14,MUTED)
    # Values must come from the saved execution, not silently edited illustration text.
    assert abs(data['c_next'][0][0]-1.1)<1e-12
    assert f"{data['h_next'][0][0]:.4f}"=='0.4803'
    for name,expected in zip('ifgo',[.25,.8,-.4,.6]):
        assert abs(data['gates_and_terms'][name][0][0]-expected)<1e-12
    return fig


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--video',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    data=json.loads((OUT/'2026-10-20.json').read_text(encoding='utf-8'))
    fig=poster(data);fig.savefig(OUT/'lstm-step.png',dpi=100);fig.savefig(OUT/'lstm-step.svg');plt.close(fig)
    stages=[]
    for stage in range(1,5):
        fig=poster(data,stage);path=OUT/f'lstm-stage-{stage}.png';fig.savefig(path,dpi=100);plt.close(fig);stages.append(path)
    if args.video:
        # Four five-second holds: this reveals arithmetic, not simulated runtime.
        concat=OUT/'animation-stages.txt'
        lines=[]
        for path in stages:lines.extend([f"file '{path.name}'",'duration 5'])
        lines.append(f"file '{stages[-1].name}'")
        concat.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y',
             '-f','concat','-safe','0','-i',str(concat),'-t','20','-vf','fps=24,format=yuv420p',
             '-c:v','libx264','-crf','20','-movflags','+faststart','-an',str(OUT/'lstm-step.mp4')]
        subprocess.run(cmd,check=True)
    print('Rendered original 1080x1350 infographic, SVG and stage frames'+(' and 20-second silent MP4' if args.video else ''))
