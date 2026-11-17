from PIL import Image,ImageOps,ImageDraw
from pathlib import Path
P=Path('/tmp/mtech_presentation_v1/package')
for page in range(2):
 c=Image.new('RGB',(1920,1620),'#e2e5e8')
 for j,i in enumerate(range(page*12+1,page*12+13)):
  im=Image.open(P/f'SLIDES/slide_{i:02}.png');im.thumbnail((960,540));c.paste(im,((j%2)*960,(j//2)*270)) if False else None
  im=Image.open(P/f'SLIDES/slide_{i:02}.png').resize((640,360));c.paste(im,((j%3)*640,(j//3)*405));ImageDraw.Draw(c).text(((j%3)*640+10,(j//3)*405+365),f'SLIDE {i}',fill='black')
 c.save(P/f'RENDERS/contact_{page+1}.png')
