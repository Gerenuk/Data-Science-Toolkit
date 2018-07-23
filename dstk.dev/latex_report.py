# -*- coding: utf-8 -*-

import os

class LatexReport:
    def __init__(self, dirname):
        self.document_start=[
r"""\documentclass{article}
\usepackage{graphicx}
\usepackage[a4paper,top=3cm, bottom=3cm, left=3cm, right=3cm]{geometry}
\usepackage{fancyhdr}
\usepackage{hyperref}
\usepackage[utf8]{inputenc}
\pagestyle{fancy}
\renewcommand{\sectionmark}[1]{\markboth{#1}{}} % set the \leftmark
\fancyhead[L]{\leftmark} % 1. sectionname
\fancyfoot[R]{\includegraphics{ThomCoLogo.png}}
\fancyfoot[L]{\includegraphics{LetsGoLogo.png}}
\rhead{\thepage}

\cfoot{}
\makeatletter
\renewcommand{\familydefault}{\sfdefault}
\renewcommand\contentsname{Inhalt}

\begin{document}
\thispagestyle{empty}
\includegraphics[width=\textwidth]{ThomcoBildTitle.png}
\tableofcontents
\newpage
"""
]
        self.document_body=[]
        self.document_end=[r"\end{document}"]

        self.dirname=dirname
        self.fignum=1

    def add_tex(self, text):
        self.document_body.append(text)

    def add_plot(self, figure, subsection=None):
        if subsection:
            self.add_subsection(subsection)

        filename="plot{}".format(self.fignum)
        self.fignum+=1
        print("Saving {}".format(filename))
        self.document_body.append("\\begin{{center}}\n"
                                  "\\includegraphics{{{}.pdf}}\n"
                                  "\\end{{center}}".format(filename))

        figure.savefig(os.path.join(self.dirname, filename+".pdf"), format="PDF", bbox_inches="tight")

    def add_section(self, name, newpage=True):
        if newpage:
            self.document_body.append(r"\newpage")
        self.document_body.append(r"\section{{{}}}".format(name))

    def add_subsection(self, name, newpage=False):
        if newpage:
            self.document_body.append(r"\newpage")
        self.document_body.append(r"\subsection{{{}}}".format(name))

    def newpage(self):
        self.document_body.append(r"\newpage")

    def write_pdf(self, filename):
        root, extension=os.path.splitext(filename)

        tex_filename=os.path.join(self.dirname, root+".tex")
        with open(tex_filename, "w", newline="\n", encoding="utf8") as texfile:
            for line in self.document_start+self.document_body+self.document_end:
                texfile.write(line+"\n")

        print("Processing {}".format(tex_filename))
        print('pdflatex "{}"'.format(tex_filename))
        #os.system('pdflatex "{}"'.format(tex_filename))
