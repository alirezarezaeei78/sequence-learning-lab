"""Original static teaching diagram from checked synthetic outputs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle


def main():
    p=Path(__file__).resolve().parent/'padding/2026-10-31'
    r=json.loads((p/'results.json').read_text(encoding='utf-8'))
    navy='#173047';teal='#087E82';amber='#AB5D13';gray='#4F6574'
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':navy,'font.size':15,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(9,12),dpi=160,facecolor='white')
    fig.text(.07,.951,'Padding is not a stop instruction',fontsize=25,weight='bold')
    fig.text(.07,.922,'A checked LSTM example · hand-set, untrained weights',fontsize=16,color=gray)
    fig.text(.07,.869,'1  PAD · add slots to make a batch',fontsize=20,weight='bold')
    ax=fig.add_axes([.07,.728,.86,.118]);ax.set_axis_off()
    for row,(name,values,valid) in enumerate([('A · length 2',[1,2,0,0],2),('B · length 4',[0,1,-1,2],4)]):
        y=.65-row*.46;ax.text(0,y+.08,name,fontsize=18,va='center')
        for j,value in enumerate(values):
            x=.31+j*.15;is_valid=j<valid
            ax.add_patch(Rectangle((x,y-.07),.13,.33,edgecolor=teal if is_valid else amber,facecolor='#E8F3F2' if is_valid else '#F8EEE3',lw=1.4))
            ax.text(x+.065,y+.10,str(value),fontsize=21,ha='center',va='center')
    fig.text(.07,.715,'B’s first zero is real data. Lengths come from the input contract.',fontsize=14,color=gray)
    fig.text(.07,.665,'2  MASK THE LOSS · choose error terms',fontsize=20,weight='bold')
    fig.text(.07,.626,'A: [1, 1, 0, 0]     B: [1, 1, 1, 1]',fontsize=21,family='DejaVu Sans Mono')
    fig.text(.07,.592,'Valid-step mean = sum(mask × squared error) / 6',fontsize=17)
    fig.text(.07,.560,'The mask excludes errors; the LSTM still processes padded inputs.',fontsize=14.5,color=amber,weight='bold')
    fig.text(.07,.508,'3  PACK · use the true sequence lengths',fontsize=20,weight='bold')
    ax=fig.add_axes([.07,.332,.86,.154]);ax.axis('off')
    rows=[['Ordinary h at padded end',f"{r['ordinary_final_h'][0]:.4f}"],['Packed h at true end',f"{r['packed_final_h'][0]:.4f}"],['Unpadded reference h',f"{r['standalone'][0]['h']:.4f}"]]
    table=ax.table(cellText=rows,colLabels=['Sequence A · final hidden state','Computed h'],cellLoc='center',colWidths=[.74,.26],loc='center')
    table.auto_set_font_size(False);table.set_fontsize(18);table.scale(1,2.45)
    for (i,j),cell in table.get_celld().items():
        cell.set_edgecolor('#CBD9DD')
        if i==0:cell.set_facecolor(navy);cell.get_text().set_color('white')
        elif i in [2,3]:cell.set_facecolor('#E8F3F2')
    fig.text(.07,.305,'Right-padded prefixes · lengths [2, 4] · enforce_sorted=False',fontsize=15,color=gray)
    fig.text(.07,.260,'TRUNCATE · remove real information',fontsize=20,weight='bold')
    fig.text(.07,.221,'B: [0, 1, −1, 2] → [0, 1], then update length to 2.',fontsize=18)
    fig.text(.07,.185,f"Final h changes: {r['packed_final_h'][1]:.4f} → {r['truncation']['B_first_two_final_h']:.4f}. This is a different input.",fontsize=17)
    fig.text(.07,.134,'Try: change only the padded suffix. Packed h should stay unchanged.',fontsize=14.5,weight='bold')
    fig.text(.07,.101,'CPU float64 · one layer/direction · input/hidden width 1 · h₀ = c₀ = 0',fontsize=13,color=gray)
    fig.text(.07,.077,'Every input/recurrent weight 0.5; input bias 0.1; recurrent bias 0.',fontsize=13,color=gray)
    fig.text(.07,.053,'Synthetic arithmetic, not predictive accuracy or speed. No bidirectional claim.',fontsize=13,color=gray)
    fig.text(.07,.024,'Source: official PyTorch 2.4 LSTM and tagged rnn.py. Original code and diagram.',fontsize=11,color=gray)
    fig.savefig(p/'padding-guide.png',dpi=160,facecolor='white');fig.savefig(p/'padding-guide.svg',facecolor='white')
    f=p/'padding-guide.svg';f.write_text('\n'.join(l.rstrip() for l in f.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
