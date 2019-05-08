#! /usr/bin/env bash

date
java -mx24g -jar supwsd-toolkit-1.0.0.jar train supconfig.xml $WSDTRAINDATA $WSDTRAINKEY
status=$?
date
exit $status
