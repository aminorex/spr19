#! /usr/bin/env bash

date
java -mx12g -jar supwsd-toolkit-1.0.0.jar test supconfig.xml $WSDTESTDATA $WSDTESTKEY
status=$?
date
exit $status
