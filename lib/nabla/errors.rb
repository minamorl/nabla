# frozen_string_literal: true

module Nabla
  class Error < StandardError; end

  # 微分器が中身を見通せないノードに触れた。定義の判らない k.let 束縛が
  # 代表例。ここで 0 を返して黙って進むことは決してしない — 勾配 0 は
  # 「その入力は結果に影響しない」という**主張**であり、知らないことを
  # 主張に変えるのは自動微分で最も見つけにくい間違いだからである。
  class Opaque < Error; end

  # 微分規則を持たない演算・扱えない文の形。
  class Unsupported < Error; end
end
