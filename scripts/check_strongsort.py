#!/usr/bin/env python
"""
StrongSortクラスのパラメータを確認するスクリプト
"""

import inspect

from boxmot.trackers.strongsort.strongsort import StrongSort

# StrongSortクラスのパラメータを取得
sig = inspect.signature(StrongSort.__init__)
parameters = sig.parameters

print("StrongSortのパラメータ:")
for name, param in parameters.items():
    if name != "self":
        default = param.default if param.default is not inspect.Parameter.empty else "必須"
        print(f"  - {name}: {default}")

# 説明文がある場合は表示
if StrongSort.__init__.__doc__:
    print("\n説明:")
    print(StrongSort.__init__.__doc__)
