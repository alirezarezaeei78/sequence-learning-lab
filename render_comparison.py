"""Original static/GIF teaching infographic; no video generation."""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

ROOT=Path('comparisons/2026-10-24')
INK='#142d4e';TEAL='#087f83';BLUE='#2861b0';AMBER='#ad6500';GRAY='#4d5e72'


def draw(stage=None):
    fig=plt.figure(figsize=(12,16),dpi=120,facecolor='white')
    ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,1200);ax.set_ylim(1600,0);ax.axis('off')
    def text(x,y,s,size=20,color=INK,weight='normal',ha='left'):
        return ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va='top',linespacing=1.35)
    def box(x,y,w,h,face='#f4f8fc',edge='#d6e1ec',lw=1.4):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01,rounding_size=14',facecolor=face,edgecolor=edge,lw=lw))
    def arrow(x1,y1,x2,y2,color):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=16,lw=2,color=color))
    text(50,35,'LSTM vs GRU',40,weight='bold')
    text(50,105,'Two ways to update a sequence state',23,color=GRAY)
    for x,title,color,sub in [(40,'LSTM',TEAL,'Carry hidden state h and cell state c'),(620,'GRU',BLUE,'Carry one hidden state h')]:
        box(x,165,540,560,face='#f6fbfb' if x==40 else '#f6f9fd')
        text(x+24,185,title,31,color=color,weight='bold')
        text(x+24,243,sub,18)
    text(68,300,'Input / forget / output gates',20,color=TEAL,weight='bold')
    text(68,341,'Candidate g can be positive or negative.',17,color=GRAY)
    box(72,398,170,70,face='white',edge=TEAL);text(157,417,'Previous c',19,ha='center')
    arrow(245,433,335,433,TEAL)
    box(339,398,205,70,face='white',edge=TEAL);text(441,417,'Updated c',19,ha='center')
    text(78,500,r'$c_t=f_t\odot c_{t-1}+i_t\odot g_t$',24)
    text(78,562,r'$h_t=o_t\odot\tanh(c_t)$',24)
    text(78,633,'Two state tensors; distinct roles.',18,color=TEAL)
    text(648,300,'Reset r / update z gates',20,color=BLUE,weight='bold')
    text(648,341,'Reset shapes the candidate calculation.',17,color=GRAY)
    box(652,398,170,70,face='white',edge=BLUE);text(737,417,'Previous h',19,ha='center')
    arrow(825,433,915,433,BLUE)
    box(919,398,205,70,face='white',edge=BLUE);text(1021,417,'Updated h',19,ha='center')
    text(658,500,r'$h_t=(1-z_t)\odot n_t+z_t\odot h_{t-1}$',22)
    text(658,565,'In this PyTorch convention:',18,color=GRAY)
    text(658,606,'z near 1 keeps more previous h.',18,color=BLUE,weight='bold')
    text(658,650,'Candidate n is not another carried state.',17,color=GRAY)
    box(40,751,1120,279,face='#fffaf0',edge='#e5cfaa')
    text(66,774,'Trace one GRU update',25,weight='bold')
    text(66,819,'Authored teaching values: previous h = 0.50, z = 0.80, candidate n = -0.25',18,color=GRAY)
    terms=[('Keep previous','0.80 × 0.50 = 0.40'),('Add candidate','0.20 × -0.25 = -0.05'),('Next hidden state','0.40 - 0.05 = 0.35')]
    for i,(label,value) in enumerate(terms):
        x=66+i*370;active=stage==i
        box(x,870,340,112,face='#e5f1ff' if active else 'white',edge=BLUE if active else '#dfd7c6',lw=3 if active else 1.2)
        text(x+16,887,label,18,color=BLUE if active else GRAY,weight='bold')
        text(x+16,928,value,18,weight='bold')
    text(66,997,'Schematic highlight loop; not training or execution time.',15,color=GRAY)
    text(50,1062,'Count parameters for the same D and H',25,weight='bold')
    text(50,1108,'D = 9, H = 16 · one layer / direction · biases on · no projection / task head',17,color=GRAY)
    box(40,1154,1120,171,face='#f7f9fc')
    text(72,1174,'Module',18,weight='bold');text(326,1174,'PyTorch parameter formula',18,weight='bold');text(1030,1174,'Count',18,weight='bold',ha='right')
    ax.plot([65,1130],[1210,1210],color='#d6e1ec',lw=1)
    text(72,1228,'LSTM',22,color=TEAL,weight='bold');text(326,1228,'4H(D + H + 2)',22);text(1030,1228,'1,728',22,ha='right',weight='bold')
    text(72,1274,'GRU',22,color=BLUE,weight='bold');text(326,1274,'3H(D + H + 2)',22);text(1030,1274,'1,296',22,ha='right',weight='bold')
    text(52,1344,'25% fewer parameters here does not establish a speed or accuracy winner.',18,color=AMBER,weight='bold')
    box(40,1401,1120,132,face='#f4f8fc')
    text(64,1418,'Implementation check',20,weight='bold')
    text(64,1458,'PyTorch GRU resets the recurrent candidate affine result, including its bias.',17)
    text(64,1493,'Reset placement and update-gate conventions can differ across implementations.',17,color=GRAY)
    text(50,1552,'Source: PyTorch 2.4.0 LSTM / GRU equations and tensor shapes. Original checked illustration.',15,color=GRAY)
    return fig


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    fig=draw();fig.savefig(ROOT/'lstm-vs-gru.png',dpi=120);fig.savefig(ROOT/'lstm-vs-gru.svg');plt.close(fig)
    svg=ROOT/'lstm-vs-gru.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    frames=[]
    for stage in [None,0,1,2,None]:
        fig=draw(stage);buffer=BytesIO();fig.savefig(buffer,format='png',dpi=100);plt.close(fig);buffer.seek(0)
        with Image.open(buffer) as im:frames.append(im.convert('RGB').copy())
    frames[0].save(ROOT/'lstm-vs-gru.gif',save_all=True,append_images=frames[1:],duration=[2500,3000,3000,3000,3500],loop=0,optimize=False)
    with Image.open(ROOT/'lstm-vs-gru.gif') as im:
        n=im.n_frames;durations=[]
        for i in range(n):im.seek(i);durations.append(im.info.get('duration',0))
        for label,i in [('first',0),('middle',n//2),('last',n-1)]:
            im.seek(i);im.convert('RGB').save(ROOT/f'gif-{label}.png')
        meta=dict(format='GIF',dimensions=list(im.size),frames=n,durations_ms=durations,total_duration_ms=sum(durations),loop=im.info.get('loop'),type='Schematic emphasis; not measured runtime',video_created=False)
    (ROOT/'media.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(meta))


if __name__=='__main__':main()
