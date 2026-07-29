# frozen_string_literal: true

require 'minitest/autorun'
require 'nabla'

module NablaTest
  # 微分の検証で「解析解と一致する」と言うためには解析解が要る。
  # 数値微分は近似なので一致の証明には使えない — 桁が合うことの確認に
  # だけ使い、厳密一致は必ず手で書いた解析解に対して主張する。
  module Helpers
    # f を x で数値微分する (中心差分)。近似であることを名前で明示する。
    def numeric_derivative(delta = 1e-6, &block)
      lambda do |x|
        (block.call(x + delta) - block.call(x - delta)) / (2 * delta)
      end
    end

    # スカラ 1 入力の式からカーネルを組む。式は param を使って組むこと。
    def scalar_kernel(name, &build)
      Moissanite.kernel(name, x: :f64) do |builder, x|
        expression = build.call(builder, x)
        builder.ret Nabla.emit(builder, expression)
      end
    end

    def assert_same_bits(expected, actual, message = nil)
      assert_equal expected, actual, message
    end
  end
end
