require 'rake/clean'

PROJECT_NAME = '2016-DLRW-brain'
DATASET = FileList[ '*/**/*.mat', '*/**/*.zip' ]
SOURCE_FILES = FileList['*/**/*.py']
DOCUMENTATION = FileList['*/**/*.tex']
#LATEXMK_OPTIONS = '-gg -d -cd -pdf -halt-on-error'
LATEXMK_OPTIONS = "-cd -pdf -gg -deps"

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
  namespace :reports do
    namespace :eeg_curiosities do
      namespace :I do
        task :compile do
          sh "latexmk #{LATEXMK_OPTIONS} doc/reports/eeg_curiosities/eeg_curiosities.tex"
        end
      end
      task I: 'I:compile'

      namespace :II do
        task :compile do
          sh "latexmk #{LATEXMK_OPTIONS} doc/reports/eeg_curiosities_followup/eeg_curiosities_followup.tex"
        end
      end
      task II: 'II:compile'

      namespace :III do
        task :compile do
          sh "latexmk #{LATEXMK_OPTIONS} doc/reports/eeg_curiosities_followup_II/eeg_curiosities_followup_II.tex"
        end
      end
      task III: 'III:compile'
    end
    task eeg_curiosities: ['eeg_curiosities:I', 'eeg_curiosities:II', 'eeg_curiosities:III']

    namespace :report do
      task :compile do
        sh "latexmk #{LATEXMK_OPTIONS} doc/reports/report.tex"
      end
    end
    task report: 'report:compile'
  end
  task reports: ['reports:report', 'reports:eeg_curiosities']

  namespace :presentation do
    task :compile do
      sh "latexmk #{LATEXMK_OPTIONS} doc/presentation/main.tex"
    end
  end
  task presentation: ['presentation:compile']

  desc "Counts words of main document"
  task :count do
    puts "#{`detex doc/main.tex | wc -w`.strip} words in project report"
  end

  desc 'Reads the report out loud'
  task :read do
    sh "detex doc/main.tex | (say --progress || espeak --stdin)"
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

  task all: [:reports, :presentation]
end

desc "Compile documents"
task doc: 'doc:all'
