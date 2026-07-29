# frozen_string_literal: true

require_relative 'lib/nabla/version'

Gem::Specification.new do |spec|
  spec.name = 'nabla'
  spec.version = Nabla::VERSION
  spec.summary = 'Symbolic automatic differentiation over moissanite expression DAGs'
  spec.description = 'Differentiate moissanite kernels symbolically at build time. Reverse mode needs ' \
                     'no runtime tape because the program is data and static, so every partial comes ' \
                     'out of one DAG traversal and compiles to native code through the same -O3 path.'
  spec.authors = ['minamorl']
  spec.email = ['minamorl@users.noreply.github.com']
  spec.license = 'MIT'
  spec.homepage = 'https://github.com/minamorl/nabla'

  spec.required_ruby_version = '>= 3.2'
  spec.metadata['rubygems_mfa_required'] = 'true'
  spec.metadata['source_code_uri'] = spec.homepage

  spec.files = Dir[
    'lib/**/*.rb',
    'README.md',
    'LICENSE'
  ]
  spec.require_paths = ['lib']

  # 式木・oracle・native backend の一切を moissanite が持つ。nabla は
  # その木に対する変換しか書かない。
  spec.add_dependency 'moissanite'
end
