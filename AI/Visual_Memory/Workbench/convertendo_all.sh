#!/bin/bash

Blue='\e[0;34m'
Light_Red='\e[1;31m'
Light_Green='\e[1;32m'
NC='\e[0m' # No Color

#font colors:
#Black				0;30		Dark_Gray			1;30
#Blue					0;34		Light_Blue		1;34
#Green				0;32		Light_Green		1;32
#Cyan					0;36		Light_Cyan		1;36
#Red					0;31		Light_Red			1;31
#Purple				0;35		Light_Purple	1;35
#Brown/Orange	0;33		Yellow				1;33
#Light Gray		0;37		White					1;37

./zerar.sh
cd iPython/
./convertendo.sh

echo -e "${Blue}Movendo arquivos${NC}"
sleep 1
if !( mv test_main.py ../../visualMemory.py ) then
	echo -e "${Light_Red}"
	read -p "Erro, precione qualquer tecla para finalizar"
	exit
fi

if !( mkdir ../../src/ )
then
	echo -e "${Light_Red}"
	read -p "Erro, precione qualquer tecla para finalizar"
	exit
fi

if !( mv Behavior.py ../../src/Behavior.py )
then
	echo -e "${Light_Red}"
	read -p "Erro, precione qualquer tecla para finalizar"
	exit
fi

if !( mkdir ../../include/ )
then
	echo -e "${Light_Red}"
	read -p "Erro, precione qualquer tecla para finalizar"
	exit
fi

if !( mv *.py ../../include/ )
then
	echo -e "${Light_Red}"
	read -p "Erro, precione qualquer tecla para finalizar"
	exit
fi

notify-send "Compilação realizada com sucesso" "Arquivos iPython compilado para python e enviados para respectivas pastas."
