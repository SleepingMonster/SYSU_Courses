#!/bin/bash 

function usage() 
{
    echo " Usage : "
    echo "   bash account_run.sh deploy"
    echo "   bash account_run.sh select account "
    echo "   bash account_run.sh select_transaction  id "
    echo "   bash account_run.sh register account asset_value "
    echo "   bash account_run.sh addTransaction id acc1 acc2 money "
    echo "   bash account_run.sh updateTransaction id money "
    echo "   bash account_run.sh splitTransaction old_id new_id acc money "
    echo "   bash account_run.sh removeTransaction id"
    echo " "
    :'
    echo " "
    echo "examples : "
    echo "   bash account_run.sh deploy "
    echo "   bash account_run.sh register  acc1  1000000 "
    echo "   bash account_run.sh register  acc2  10000 "
    echo "   bash account_run.sh register  acc3  10000 "
    echo "   bash account_run.sh register  bank  1000000000 "
    echo "   bash account_run.sh select acc1"
    echo "   bash account_run.sh select acc2"
    echo "   bash account_run.sh addTransaction 0001 acc1 acc2 100"
    echo "   bash account_run.sh select_transaction 0001"
    echo "   bash account_run.sh splitTransaction 0001 0002 acc3 50"
    echo "   bash account_run.sh splitTransaction 0001 0003 bank 10"
    echo "   bash account_run.sh select_transaction 0002"
    echo "   bash account_run.sh select_transaction 0003"
    echo "   bash account_run.sh updateTransaction 0001 40"
    echo "   bash account_run.sh updateTransaction 0002 50"
    echo "   bash account_run.sh updateTransaction 0003 10"
    '
    exit 0
}

    case $1 in
    deploy)
            [ $# -lt 1 ] && { usage; }
            ;;
    select)
            [ $# -lt 2 ] && { usage; }
            ;;
    select_transaction)
            [ $# -lt 2 ] && { usage; }
            ;;
    register)
            [ $# -lt 3 ] && { usage; }
            ;;
    addTransaction)
            [ $# -lt 5 ] && { usage; }
            ;;
    splitTransaction)
            [ $# -lt 5 ] && { usage; }
            ;;
    updateTransaction)
            [ $# -lt 3 ] && { usage; }
            ;;
    removeTransaction)
            [ $# -lt 2 ] && { usage; }
            ;;
    *)
        usage
            ;;
    esac

    java -Djdk.tls.namedGroups="secp256k1" -cp 'apps/*:conf/:lib/*' org.fisco.bcos.account.client.AccountClient $@

