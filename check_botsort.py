#!/usr/bin/env python
"""
BotSortクラスのパラメータを確認するスクリプト
"""

import inspect
from boxmot.trackers.botsort.botsort import BotSort

# BotSortクラスのパラメータを取得
sig = inspect.signature(BotSort.__init__)
parameters = sig.parameters

print("BotSortのパラメータ:")
for name, param in parameters.items():
    if name != 'self':
        default = param.default if param.default is not inspect.Parameter.empty else "必須"
        print(f"  - {name}: {default}")

# 説明文がある場合は表示
if BotSort.__init__.__doc__:
    print("\n説明:")
    print(BotSort.__init__.__doc__) 