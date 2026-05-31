from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, ViTModel


class KhoiXuLyVanBan(nn.Module):
    """Encoder Transformer cho văn bản + lớp giảm chiều."""

    def __init__(self, ten_mo_hinh: str, kich_thuoc_an: int, so_tang_dong_bang: int = 0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(ten_mo_hinh)
        self._dong_bang_tang(self.encoder, so_tang_dong_bang)
        kich_thuoc_an_ra = self.encoder.config.hidden_size
        self.giam_chieu = nn.Linear(kich_thuoc_an_ra, kich_thuoc_an)
        self.chuan_hoa_lo = nn.BatchNorm1d(kich_thuoc_an)

    @staticmethod
    def _dong_bang_tang(model: nn.Module, so_tang: int) -> None:
        if so_tang <= 0:
            return
        layers = None
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            layers = model.encoder.layer
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
            layers = model.encoder.layers
        if layers is None:
            return
        for idx, layer in enumerate(layers):
            if idx < so_tang:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        dau_vao = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        if token_type_ids is not None:
            dau_vao['token_type_ids'] = token_type_ids
        dau_ra = self.encoder(**dau_vao)
        bieu_dien_cls = dau_ra.last_hidden_state[:, 0, :]
        bieu_dien_giam = F.leaky_relu(self.giam_chieu(bieu_dien_cls))
        return self.chuan_hoa_lo(bieu_dien_giam)


class KhoiXuLyAnh(nn.Module):
    """Encoder ViT cho ảnh + lớp giảm chiều."""

    def __init__(self, ten_mo_hinh: str, kich_thuoc_an: int, so_tang_dong_bang: int = 0):
        super().__init__()
        self.encoder = ViTModel.from_pretrained(ten_mo_hinh)
        self._dong_bang_tang(self.encoder, so_tang_dong_bang)
        kich_thuoc_an_ra = self.encoder.config.hidden_size
        self.giam_chieu = nn.Linear(kich_thuoc_an_ra, kich_thuoc_an)
        self.chuan_hoa_lo = nn.BatchNorm1d(kich_thuoc_an)

    @staticmethod
    def _dong_bang_tang(model: nn.Module, so_tang: int) -> None:
        if so_tang <= 0:
            return
        layers = None
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            layers = model.encoder.layer
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
            layers = model.encoder.layers
        if layers is None:
            return
        for idx, layer in enumerate(layers):
            if idx < so_tang:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, pixel_values):
        dau_ra = self.encoder(pixel_values=pixel_values)
        bieu_dien_cls = dau_ra.last_hidden_state[:, 0, :]
        bieu_dien_giam = F.leaky_relu(self.giam_chieu(bieu_dien_cls))
        return self.chuan_hoa_lo(bieu_dien_giam)


class DT4MID(nn.Module):
    """
    Dual Transformer for Multimodal Irony Detection.

    Kiến trúc bám theo ý tưởng DT4MID trong bài báo:
    - text transformer encoder
    - vision transformer encoder
    - projection + LeakyReLU + BatchNorm cho từng nhánh
    - early fusion bằng phép nối vector
    - MLP classifier cho bài toán nhị phân

    Đây là bản tích hợp cho pipeline hiện tại với 1 văn bản + 1 ảnh.
    """

    def __init__(
        self,
        ten_mo_hinh_chu: str = 'vinai/phobert-base',
        ten_mo_hinh_anh: str = 'google/vit-base-patch16-224-in21k',
        kich_thuoc_an: int = 64,
        kich_thuoc_lop_an: int = 32,
        ti_le_dropout: float = 0.3,
        so_nhan: int = 2,
        chi_dung_van_ban: bool = False,
        chi_dung_anh: bool = False,
        dong_bang_tang_van_ban: int = 0,
        dong_bang_tang_anh: int = 0,
    ):
        super().__init__()
        if chi_dung_van_ban and chi_dung_anh:
            raise ValueError('Khong the dong thoi bat ca chi_dung_van_ban va chi_dung_anh.')

        self.chi_dung_van_ban = chi_dung_van_ban
        self.chi_dung_anh = chi_dung_anh

        if not chi_dung_anh:
            self.xu_ly_van_ban = KhoiXuLyVanBan(
                ten_mo_hinh_chu,
                kich_thuoc_an,
                so_tang_dong_bang=dong_bang_tang_van_ban,
            )
        if not chi_dung_van_ban:
            self.xu_ly_anh = KhoiXuLyAnh(
                ten_mo_hinh_anh,
                kich_thuoc_an,
                so_tang_dong_bang=dong_bang_tang_anh,
            )

        if chi_dung_van_ban or chi_dung_anh:
            kich_thuoc_ghep = kich_thuoc_an
        else:
            kich_thuoc_ghep = kich_thuoc_an * 2

        self.hop_nhat = nn.Linear(kich_thuoc_ghep, kich_thuoc_an)
        self.dropout = nn.Dropout(ti_le_dropout)
        self.lop_an = nn.Linear(kich_thuoc_an, kich_thuoc_lop_an)
        self.dau_ra = nn.Linear(kich_thuoc_lop_an, so_nhan)

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, token_type_ids=None):
        cac_bieu_dien = []

        if not self.chi_dung_anh:
            if input_ids is None or attention_mask is None:
                raise ValueError('DT4MID can input_ids va attention_mask cho nhanh van ban.')
            bieu_dien_chu = self.xu_ly_van_ban(input_ids, attention_mask, token_type_ids=token_type_ids)
            cac_bieu_dien.append(bieu_dien_chu)

        if not self.chi_dung_van_ban:
            if pixel_values is None:
                raise ValueError('DT4MID can pixel_values cho nhanh anh.')
            bieu_dien_anh = self.xu_ly_anh(pixel_values)
            cac_bieu_dien.append(bieu_dien_anh)

        if not cac_bieu_dien:
            raise ValueError('DT4MID khong nhan duoc dau vao hop le nao.')

        h0 = cac_bieu_dien[0] if len(cac_bieu_dien) == 1 else torch.cat(cac_bieu_dien, dim=-1)
        h1 = F.leaky_relu(self.hop_nhat(h0))
        h2 = self.dropout(h1)
        h3 = F.relu(self.lop_an(h2))
        return self.dau_ra(h3)
