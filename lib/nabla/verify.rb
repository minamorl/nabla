# frozen_string_literal: true

module Nabla
  # ==================================================================
  # Verify — 積分の答えを微分して問いに戻れるか確かめる。
  #
  # **この層の存在理由**である。表引きの CAS は答えを返して終わりで、
  # 正しさは実装への信頼に帰着する。ここには微分器があるので、
  # ∫f = F の主張はそのまま F' = f という**検査可能な命題**になる。
  #
  # 二段構えで、二段目に落ちたことを隠さない:
  #
  #   :symbolic — F' を簡約したら f と構文的に一致した。微分規則が
  #               正しい限りこの積分は正しい。数値は一切関与しない。
  #               証明書と呼べるのはこの段だけである。
  #   :numeric  — 一致しなかったので、両者を native カーネルにして
  #               乱択した点で突き合わせた。反例が無いことしか言えない。
  #
  # 記号一致率は簡約器の強さの関数なので、そのまま開発の指標になる。
  # ==================================================================
  module Verify
    Result = Data.define(:status, :max_relative_error, :samples) do
      def symbolic? = status == :symbolic
      def ok?(tolerance = 1e-9) = symbolic? || max_relative_error <= tolerance

      def to_s
        symbolic? ? 'symbolically exact' : format('numeric agreement, max relative error %.2e', max_relative_error)
      end
    end

    module_function

    # 既定の標本域は 0 と負数を避ける (log や 1/x の定義域の外で
    # 突き合わせても、積分が間違っている証拠にはならないため)。
    def integral(integrand, primitive, variable, samples: 2000, range: 0.31..3.7, seed: 11)
      differentiated = Simplify.run(derivative(primitive, variable))
      return Result.new(status: :symbolic, max_relative_error: 0.0, samples: 0) if
        differentiated == Simplify.run(integrand)

      sampling = { samples: samples, range: range, seed: seed }
      worst = compare_numerically(integrand, differentiated, variable, sampling)
      Result.new(status: :numeric, max_relative_error: worst, samples: samples)
    end

    def derivative(primitive, variable)
      Forward.derivative(primitive, variable)
    end

    def compare_numerically(integrand, differentiated, variable, sampling)
      wanted = compile(integrand, variable)
      got = compile(differentiated, variable)
      random = Random.new(sampling.fetch(:seed))
      range = sampling.fetch(:range)
      span = range.end - range.begin
      worst = 0.0
      sampling.fetch(:samples).times do
        point = range.begin + (random.rand * span)
        expected = wanted.call(point)
        next if expected.zero? || !expected.finite?

        worst = [worst, ((got.call(point) - expected) / expected).abs].max
      end
      worst
    end

    def compile(expression, variable)
      Moissanite.kernel(:verify_point, x: :f64) do |builder, x|
        builder.ret Nabla.emit(builder, Rewrite.substitute(expression, variable, x))
      end
    end
  end
end
