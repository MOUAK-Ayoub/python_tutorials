pipeline {
  agent any
  stages {
    stage('echo  msg') {
      parallel {
        stage('echo  msg') {
          steps {
            echo 'test'
          }
        }

        stage('msg  3') {
          steps {
            echo 'test0'
          }
        }

      }
    }

    stage('next msg') {
      steps {
        echo 'test 2'
      }
    }

  }
}