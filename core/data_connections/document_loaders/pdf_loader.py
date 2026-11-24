"""Loader that loads image files."""
from typing import List, Optional

import os
import fitz
from langchain_community.document_loaders import UnstructuredFileLoader
from unstructured.partition.text import partition_text


class UnstructuredPaddlePDFLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load image files, such as PNGs and JPGs."""

    def __init__(self, file_path: str, **kwargs):
        """
        Args:
            file_path: The path to the image file.
            **kwargs: Keyword arguments to pass to unstructured.
        """
        self.file_path = file_path
        self.is_chinese_text = kwargs.pop("is_chinese_text", True)
        super().__init__(file_path, **kwargs)

    def _get_elements(self) -> List:
        def pdf_ocr_txt(filepath, dir_path="tmp_files"):
            from paddleocr import PaddleOCR

            full_dir_path = os.path.join(os.path.dirname(filepath), dir_path)
            if not os.path.exists(full_dir_path):
                os.makedirs(full_dir_path)

            if self.is_chinese_text:
                lang = "ch"
            else:
                lang = "en"
            ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=False, show_log=False)
            doc = fitz.open(filepath)
            txt_file_path = os.path.join(full_dir_path, f"{os.path.split(filepath)[-1]}.txt")
            img_name = os.path.join(full_dir_path, 'tmp.png')
            with open(txt_file_path, 'w', encoding='utf-8') as fout:
                for i in range(doc.page_count):
                    page = doc[i]
                    text = page.get_text("")
                    fout.write(text)
                    fout.write("\n")

                    img_list = page.get_images()
                    for img in img_list:
                        pix = fitz.Pixmap(doc, img[0])
                        if pix.n - pix.alpha >= 4:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        pix.save(img_name)

                        result = ocr.ocr(img_name)
                        ocr_result = [i[1][0] for line in result for i in line]
                        fout.write("\n".join(ocr_result))
            if os.path.exists(img_name):
                os.remove(img_name)
            return txt_file_path

        txt_file_path = pdf_ocr_txt(self.file_path)
        return partition_text(filename=txt_file_path,
                              file=self.unstructured_kwargs.get("file", None),
                              text=self.unstructured_kwargs.get("text", None),
                              encoding=self.unstructured_kwargs.get("encoding", "utf-8"),
                              paragraph_grouper=self.unstructured_kwargs.get("paragraph_grouper", None))


if __name__ == "__main__":
    filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "content", "samples", "test.pdf")
    loader = UnstructuredPaddlePDFLoader(filepath, mode="single")
    docs = loader.load()
    for doc in docs:
        print(doc)
