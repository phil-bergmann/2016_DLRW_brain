require 'rake/clean'

PROJECT_NAME = '2016-DLRW-brain'
DATASET = FileList[ '*/**/*.mat', '*/**/*.zip' ]
SOURCE_FILES = FileList['*/**/*.py']
DOCUMENTATION = FileList['*/**/*.tex']

CLEAN << FileList['doc/*.{aux,log,out}']
CLEAN << DATASET

desc "Grep out the TODO's"
task :todo do
  puts "\n** Whats left to do for #{PROJECT_NAME} **\n"
  puts `grep -n TODO */*.py */*.tex`
end

desc 'install requirements'
task :init do
  puts `pip install -r requirements.txt`
end

desc 'run pylint'
task :lint do
  puts `pylint brain`
end

desc 'run tests'
task :test do
  puts `nosetests`
end

namespace :doc do
  namespace :compile do
    task :once do
    `latexmk -gg -d -cd -pv -pdf -halt-on-error -jobname=#{PROJECT_NAME} doc/main.tex`
    end
    
    desc 'countinuusly run latexmk'
    task :continuous do
    `latexmk -gg -d -cd -pvc -pdf -halt-on-error -jobname=#{PROJECT_NAME} doc/main.tex`
    end
  end

  desc 'compile the report using latexmk'
  task compile: 'compile:once'

  desc "Counts words of main document"
  task :count do
    puts "#{`detex doc/main.tex | wc -w`.strip} words in project report"
  end

  desc "Count PDF Pages"
  task :pages do
    puts "Pages for Project #{PROJECT_NAME}:"
    puts `pdfinfo doc/#{PROJECT_NAME}.pdf|grep Pages`
  end

  task :open do
    puts "opening #{PROJECT_NAME}"
    `open doc/#{PROJECT_NAME}.pdf`
  end

  task all: [:count, :pages, :compile]
end

task doc: 'doc:all'
