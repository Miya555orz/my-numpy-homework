"""生成4页深圳大学Python项目实验报告。"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'output'/'pdf'/'project1_experiment_report.pdf'
OUTPUT.parent.mkdir(parents=True,exist_ok=True)

pdfmetrics.registerFont(TTFont('YaHei',r'C:\Windows\Fonts\simsun.ttc'))
pdfmetrics.registerFont(TTFont('YaHeiBold',r'C:\Windows\Fonts\simsun.ttc'))
pdfmetrics.registerFont(TTFont('Consolas',r'C:\Windows\Fonts\consola.ttf'))

PAGE_W,PAGE_H=A4
BLUE=colors.HexColor('#005BAC')
LIGHT_BLUE=colors.HexColor('#EAF3FB')
INK=colors.HexColor('#1F2937')
MUTED=colors.HexColor('#5B6573')
LINE=colors.HexColor('#CBD5E1')
GREEN=colors.HexColor('#16794A')


body_style=ParagraphStyle(
    'body',fontName='YaHei',fontSize=9.2,leading=14,
    textColor=INK,alignment=TA_JUSTIFY,wordWrap='CJK',
)
small_style=ParagraphStyle(
    'small',fontName='YaHei',fontSize=8,leading=11.5,
    textColor=MUTED,alignment=TA_LEFT,wordWrap='CJK',
)
table_style=ParagraphStyle(
    'table',fontName='YaHei',fontSize=7.6,leading=10,
    textColor=INK,alignment=TA_LEFT,wordWrap='CJK',
)


def header_footer(c,page):
    c.setStrokeColor(LINE)
    c.setLineWidth(0.6)
    c.line(18*mm,PAGE_H-16*mm,PAGE_W-18*mm,PAGE_H-16*mm)
    c.setFont('YaHei',8)
    c.setFillColor(MUTED)
    c.drawString(18*mm,PAGE_H-12.5*mm,'深圳大学实验报告 | Python程序设计')
    c.drawRightString(PAGE_W-18*mm,11*mm,f'第 {page} 页 / 共 4 页')


def section(c,number,title,y):
    c.setFillColor(BLUE)
    c.rect(18*mm,y-5.5*mm,3*mm,5.5*mm,fill=1,stroke=0)
    c.setFont('YaHeiBold',13)
    c.drawString(24*mm,y-4.3*mm,f'{number}  {title}')
    return y-10*mm


def paragraph(c,text,y,width=174*mm,style=body_style,x=18*mm):
    p=Paragraph(text,style)
    _,height=p.wrap(width,PAGE_H)
    p.drawOn(c,x,y-height)
    return y-height


def info_line(c,label,value,y):
    c.setFont('YaHeiBold',12)
    c.setFillColor(INK)
    c.drawString(46*mm,y,label)
    c.setStrokeColor(colors.HexColor('#64748B'))
    c.line(82*mm,y-1.5*mm,163*mm,y-1.5*mm)
    c.setFont('YaHei',11)
    c.drawCentredString(122.5*mm,y+0.5*mm,value)


def flow_box(c,x,y,w,h,title,subtitle):
    c.setStrokeColor(BLUE)
    c.setFillColor(LIGHT_BLUE)
    c.roundRect(x,y,w,h,2*mm,fill=1,stroke=1)
    c.setFillColor(INK)
    c.setFont('YaHeiBold',9)
    c.drawCentredString(x+w/2,y+h-6*mm,title)
    c.setFillColor(MUTED)
    c.setFont('YaHei',7.2)
    c.drawCentredString(x+w/2,y+4*mm,subtitle)


def arrow(c,x1,y,x2):
    c.setStrokeColor(colors.HexColor('#64748B'))
    c.setFillColor(colors.HexColor('#64748B'))
    c.setLineWidth(1)
    c.line(x1,y,x2,y)
    c.line(x2,y,x2-2.3*mm,y+1.5*mm)
    c.line(x2,y,x2-2.3*mm,y-1.5*mm)


def code_block(c,lines,x,y,w):
    line_h=4.2*mm
    h=(len(lines)+1)*line_h
    c.setFillColor(colors.HexColor('#F7F9FC'))
    c.setStrokeColor(LINE)
    c.roundRect(x,y-h,w,h,2*mm,fill=1,stroke=1)
    c.setFillColor(colors.HexColor('#243447'))
    c.setFont('Consolas',7.2)
    cursor=y-5*mm
    for line in lines:
        c.drawString(x+4*mm,cursor,line)
        cursor-=line_h
    return y-h


c=canvas.Canvas(str(OUTPUT),pagesize=A4)
c.setTitle('Project 1 NumPy图像编码器实验报告')
c.setAuthor('郑心而')

# 第1页：封面
c.setFillColor(BLUE)
c.rect(0,PAGE_H-12*mm,PAGE_W,12*mm,fill=1,stroke=0)
c.setFillColor(INK)
c.setFont('YaHeiBold',27)
c.drawCentredString(PAGE_W/2,PAGE_H-48*mm,'深 圳 大 学 实 验 报 告')
c.setFillColor(BLUE)
c.setFont('YaHeiBold',16)
c.drawCentredString(PAGE_W/2,PAGE_H-65*mm,'Project 1: NumPy图像编码器')
c.setFillColor(MUTED)
c.setFont('YaHei',10)
c.drawCentredString(PAGE_W/2,PAGE_H-73*mm,'OpenCLIP ViT-B-32 Image Encoder Forward in NumPy')

cover_y=PAGE_H-100*mm
info_line(c,'课程名称','Python程序设计',cover_y)
info_line(c,'项目名称','NumPy图像编码器',cover_y-13*mm)
info_line(c,'学    院','人工智能学院',cover_y-26*mm)
info_line(c,'专    业','人工智能',cover_y-39*mm)
info_line(c,'指导教师','樊超',cover_y-52*mm)
info_line(c,'报告人','郑心而',cover_y-65*mm)
info_line(c,'学    号','2025300053',cover_y-78*mm)
info_line(c,'实验时间','2026.11.17 - 2026.11.18',cover_y-91*mm)
info_line(c,'提交时间','2026.11.18',cover_y-104*mm)

c.setFillColor(LIGHT_BLUE)
c.roundRect(32*mm,31*mm,146*mm,18*mm,2*mm,fill=1,stroke=0)
c.setFillColor(BLUE)
c.setFont('YaHeiBold',9)
c.drawString(38*mm,42*mm,'代码仓库')
c.setFillColor(INK)
c.setFont('YaHei',7.5)
c.drawString(38*mm,35.5*mm,'https://github.com/Miya555orz/my-numpy-homework')
c.setFillColor(MUTED)
c.setFont('YaHei',8)
c.drawCentredString(PAGE_W/2,18*mm,'教务处制')
c.showPage()

# 第2页：目标与方法
header_footer(c,2)
y=PAGE_H-25*mm
y=section(c,'1','实验目标',y)
y=paragraph(c,'本实验要求只使用 NumPy 完成 OpenCLIP ViT-B-32 图像编码器的前向计算。输入是预处理后的 <b>[B,3,224,224]</b> 浮点图像，输出是每张图对应的 <b>[B,512]</b> 单位向量。实验重点是理解数组 shape、广播、矩阵乘法、多头注意力、残差连接和批量计算，并使结果在给定容差内匹配独立 FP32 参考。',y)

y-=7*mm
y=section(c,'2','总体流程',y)
box_y=y-28*mm
box_w=28*mm
gap=7*mm
start=18*mm
items=[
    ('图像输入','B x 3 x 224 x 224'),
    ('Patch','B x 49 x 3072'),
    ('Token','B x 50 x 768'),
    ('12层Block','B x 50 x 768'),
    ('CLS输出','B x 512'),
]
for i,(title,subtitle) in enumerate(items):
    x=start+i*(box_w+gap)
    flow_box(c,x,box_y,box_w,20*mm,title,subtitle)
    if i<len(items)-1:
        arrow(c,x+box_w+1*mm,box_y+10*mm,x+box_w+gap-1*mm)
y=box_y-9*mm
y=paragraph(c,'数据流可以概括为：先把图像分成 7 x 7 个 patch，每个 patch 展开为 3072 个数；再投影到 768 维，前面加入 CLS token 并添加位置编码；经过 12 个 Transformer block 后只取 CLS 行，最终投影到 512 维并做单位长度归一化。',y)

y-=7*mm
y=section(c,'2.1','一个关键 shape 变化',y)
shape_data=[
    [Paragraph('<b>步骤</b>',table_style),Paragraph('<b>操作</b>',table_style),Paragraph('<b>shape</b>',table_style)],
    ['输入','原始批量图像','[B,3,224,224]'],
    ['拆轴','reshape出patch网格','[B,3,7,32,7,32]'],
    ['换轴','网格轴移到通道轴前','[B,7,7,3,32,32]'],
    ['展开','49个patch分别展平','[B,49,3072]'],
]
t=Table(shape_data,colWidths=[25*mm,78*mm,61*mm],rowHeights=7.5*mm)
t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),BLUE),('TEXTCOLOR',(0,0),(-1,0),colors.white),
    ('FONTNAME',(0,0),(-1,-1),'YaHei'),('FONTSIZE',(0,0),(-1,-1),8),
    ('GRID',(0,0),(-1,-1),0.5,LINE),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ('BACKGROUND',(0,1),(-1,-1),colors.white),('LEFTPADDING',(0,0),(-1,-1),4),
]))
_,th=t.wrap(164*mm,80*mm)
t.drawOn(c,24*mm,y-th)
y-=th+5*mm
y=paragraph(c,'其中 transpose(0,2,4,1,3,5) 保证 patch 按“从左到右、从上到下”排列，同时每个 patch 内部仍按 C、H、W 顺序展开。',y,width=164*mm,x=24*mm)
c.showPage()

# 第3页：实现与易错点
header_footer(c,3)
y=PAGE_H-25*mm
y=section(c,'3','关键实现',y)
y=paragraph(c,'实现保持每一步 shape 清楚可查：注意力先一次性生成 Q、K、V，再拆成 12 个 head；每个 head 的维度为 64。注意力系数用于混合 V，随后把 head 轴合并回 768 维。Transformer block 中包含两次残差连接，MLP 中间宽度为 3072。',y)

y-=5*mm
c.setFont('YaHeiBold',10)
c.setFillColor(INK)
c.drawString(24*mm,y,'多头注意力核心')
y-=4*mm
y=code_block(c,[
    'q = split_heads(qkv[:, :, :width])',
    'k = split_heads(qkv[:, :, width:2*width])',
    'scores = (q @ k.transpose(0,1,3,2)) / sqrt(64)',
    'context = attention_weights(scores) @ v',
    'merged = context.transpose(0,2,1,3).reshape(B,N,768)',
],24*mm,y,162*mm)

y-=7*mm
c.setFont('YaHeiBold',10)
c.drawString(24*mm,y,'残差连接与MLP')
y-=4*mm
y=code_block(c,[
    'residual = x + attention(norm1(x))',
    'u = norm2(residual)',
    'hidden = GELU(u @ W1.T + b1)',
    'output = residual + hidden @ W2.T + b2',
],24*mm,y,162*mm)

y-=7*mm
y=section(c,'3.1','易错点与修正',y)
y=paragraph(c,'<b>K 的转置：</b>课堂中的二维示例可以写 <font name="Consolas">k.T</font>，但真实 K 是四维数组 [B,H,N,64]。如果直接使用 <font name="Consolas">.T</font>，NumPy 会反转全部轴，破坏 batch 和 head 的组织。实现中改为 <font name="Consolas">k.transpose(0,1,3,2)</font>，只交换最后两个轴，得到 [B,H,64,N]，因此 scores 正确得到 [B,H,N,N]。',y)
y-=4*mm
y=paragraph(c,'<b>权重方向：</b>注意力和 MLP 的线性权重按 [out,in] 保存，因此使用 <font name="Consolas">weight.T</font>；最后的 visual.proj 已经是 [in,out]=[768,512]，不能再次转置。分阶段检查可以在错误扩散到 12 层之前定位问题。',y)

y-=6*mm
y=section(c,'3.2','NumPy练习的作用',y)
y=paragraph(c,'12 道小任务覆盖了切片、点积、向量长度、广播、axis、图像 batch、余弦检索、距离矩阵和小型前向计算。它们分别对应编码器中的 patch 组织、注意力点积、归一化、批量矩阵运算和 MLP，是主项目实现前的最小验证。',y)
c.showPage()

# 第4页：结果与讨论
header_footer(c,4)
y=PAGE_H-25*mm
y=section(c,'4','实验结果',y)
result_rows=[
    ['阶段','输出shape','最大绝对误差','平均绝对误差','结论'],
    ['preprocess','(10,3,224,224)','0.000e+00','0.000e+00','PASS'],
    ['patches','(2,49,3072)','0.000e+00','0.000e+00','PASS'],
    ['tokens','(2,50,768)','4.005e-05','1.289e-06','PASS'],
    ['block0','(2,50,768)','7.063e-06','3.250e-07','PASS'],
    ['last_tokens','(2,50,768)','4.768e-05','2.696e-06','PASS'],
    ['output','(10,512)','5.439e-07','8.311e-08','PASS'],
    ['batch','(3,512)','4.619e-07','7.632e-08','PASS'],
]
t=Table(result_rows,colWidths=[28*mm,43*mm,36*mm,36*mm,21*mm],rowHeights=7*mm)
t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),BLUE),('TEXTCOLOR',(0,0),(-1,0),colors.white),
    ('FONTNAME',(0,0),(-1,-1),'YaHei'),('FONTSIZE',(0,0),(-1,-1),7.5),
    ('GRID',(0,0),(-1,-1),0.45,LINE),('ALIGN',(1,1),(-2,-1),'CENTER'),
    ('TEXTCOLOR',(-1,1),(-1,-1),GREEN),('FONTNAME',(-1,1),(-1,-1),'YaHeiBold'),
    ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
]))
_,th=t.wrap(164*mm,100*mm)
t.drawOn(c,24*mm,y-th)
y-=th+7*mm
y=paragraph(c,'全部阶段均通过独立 FP32 参考检查。最终 output 最大绝对误差为 5.439e-07，小于 atol=2e-5；乱序且含重复图像的 batch 检查也通过。最终每个 512 维输出的平方和接近 1。',y)

y-=6*mm
y=section(c,'5','讨论与总结',y)
y=paragraph(c,'实验表明，ViT 图像塔的核心可以用基础 NumPy 运算完整表达。最重要的不是记住公式，而是持续跟踪每一步 shape：patchify 改变数据组织方式，多头注意力拆分和恢复特征轴，残差连接保持主干 shape 不变。分阶段数值检查能够快速区分 patch 顺序、权重转置、注意力轴和最终投影等问题。当前实现的主要性能限制是课程提供的精确 GELU 使用 Python 标量循环，因此速度慢于向量化框架，但数值结果满足任务要求。',y)

y-=5*mm
y=section(c,'6','问题修正与工作说明',y)
y=paragraph(c,'复核练习合并文件时，发现 E5 的函数名与题目要求不一致，已统一为 unit_vector 并重新运行。M1 的独立版本缺少零查询向量检查，现已在分母计算前加入 ValueError。本人负责练习实现、编码器代码核对和结果验证。报告整理使用课程 starter、检查器及 OpenAI Codex 辅助；数值结果以本地检查器输出为依据。',y)

y-=5*mm
c.setFillColor(INK)
c.setFont('YaHei',9.2)
c.drawString(18*mm,y-8*mm,'复现命令：python check.py --stage all')

c.save()
print('Saved',OUTPUT)
