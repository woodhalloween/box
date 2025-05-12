#!/usr/bin/env python
"""
boxmotのインポート可能なモジュールとクラスを確認するスクリプト
"""

import os
import pkgutil
import importlib

# boxmotパッケージのルートを取得
import boxmot


def explore_package(package_name, prefix=""):
    """パッケージの構造を探索し、インポート可能なモジュールを出力する"""
    package = importlib.import_module(package_name)

    print(f"\n{prefix}パッケージ: {package_name}")

    try:
        path = package.__path__
    except AttributeError:
        print(f"{prefix}  Not a package")
        return

    # モジュールを列挙
    for loader, name, is_pkg in pkgutil.iter_modules(path):
        full_name = f"{package_name}.{name}"

        # モジュールをインポート
        try:
            module = importlib.import_module(full_name)
            if is_pkg:
                print(f"{prefix}  サブパッケージ: {name}")
                explore_package(full_name, prefix + "    ")
            else:
                print(f"{prefix}  モジュール: {name}")

                # モジュール内のクラスを探す
                import inspect

                classes = [
                    cls_name
                    for cls_name, cls_obj in inspect.getmembers(module)
                    if inspect.isclass(cls_obj) and cls_obj.__module__ == full_name
                ]

                if classes:
                    print(f"{prefix}    クラス: {', '.join(classes)}")
        except Exception as e:
            print(f"{prefix}  エラー: {full_name} - {str(e)}")


if __name__ == "__main__":
    explore_package("boxmot")
