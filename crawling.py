from pdfminer.pdfinterp import PDFResourceManager, process_pdf # pip install
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO
from io import open
from urllib.request import urlopen
import codecs
import csv

def read_pdf_file(pdfFile):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)

    process_pdf(rsrcmgr, device, pdfFile)
    device.close()

    content = retstr.getvalue()
    retstr.close()
    return content

f = open('20191122_link.csv','r',encoding = 'utf-8')
rdr = csv.reader(f)

count = 0
for line in rdr:
    lines = ''.join(line)
    n_line = lines.replace('https://arxiv.org/pdf/','')
    count += 1
    file = codecs.open(n_line + ".txt", "a", encoding='utf-8')
    line_txt = line[0] + ".pdf"
    pdf_file = urlopen(line_txt)
    contents = read_pdf_file(pdf_file)
    file.write(contents)
    file.close()

file.close()
f.close()












