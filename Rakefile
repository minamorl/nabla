# frozen_string_literal: true

require 'rake/testtask'
require 'rubocop/rake_task'

Rake::TestTask.new(:test) do |task|
  task.libs << 'test'
  task.pattern = 'test/**/*_test.rb'
end

RuboCop::RakeTask.new(:rubocop)

desc 'Run every example (they assert their own invariants, so this is a test too)'
task :examples do
  Dir.glob('examples/*.rb').each { |path| ruby path }
end

task lint: :rubocop
task default: %i[test rubocop]
