require 'rake/clean'

PROJECT_NAME = '2016-DLRW-brain'
DATASET = FileList[ 'data/**/*.mat, data/**/*.zip' ]
SOURCE_FILES = FileList['*/**/*.py']
DOCUMENTATION = FileList['*/**/*.tex']

CLEAN << DATASET
CLEAN << FileList['doc/*.{aux,log,out}']

desc "Grep out the TODO's"
task :todo do
  puts "\n** Whats left to do for #{PROJECT_NAME} **\n"
  puts `grep -n TODO */*.py */*.tex`
end

desc 'install requirements'
task :init do
  puts `pip install -r requirements.txt`
end

task :lint do
  puts `pylint data`
end

task :test do
  puts `nosetests`
end

namespace :doc do
  task all: [:compile]
  task :compile do
    puts "Compiling report for #{PROJECT_NAME}"
    `pdflatex -output-directory=doc -halt-on-error -jobname=#{PROJECT_NAME} doc/main.tex`
  end

  desc "Counts words of main document"
  task :count do
    puts "#{`detex doc/main.tex | wc -w`.strip} words in project report"
  end

  desc "Count PDF Pages"
  task :pages do
    puts "Pages for Project #{PROJECT_NAME}:"
    puts `pdfinfo doc/#{PROJECT_NAME}.pdf|grep Pages`
  end
end

task doc: 'doc:all'
