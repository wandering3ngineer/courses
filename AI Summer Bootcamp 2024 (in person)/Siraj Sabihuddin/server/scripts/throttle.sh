#!/bin/bash

INTERFACE="eth0"
IFB_INTERFACE="ifb0"
RATE="8mbit"
BURST="32kbit"
LATENCY="300ms"

function show_tc {
    local iface=$1
    echo "Current qdisc rules for $iface:"
    sudo tc qdisc show dev $iface

    echo "Extracting current settings for $iface:"
    sudo tc -s qdisc show dev $iface | grep -E "qdisc|limit|rate|burst|latency|target|interval|memory_limit|quantum"
    echo ""
}

function apply_tc {
    echo "Applying traffic control rules..."

    # Show current state
    show_tc $INTERFACE
    show_tc $IFB_INTERFACE

    # Load the IFB module and set up the IFB interface
    sudo modprobe ifb
    sudo ip link add $IFB_INTERFACE type ifb
    sudo ip link set $IFB_INTERFACE up
    
    # Apply outgoing bandwidth limit
    sudo tc qdisc add dev $INTERFACE root tbf rate $RATE burst $BURST latency $LATENCY
    
    # Redirect incoming traffic to IFB and apply incoming bandwidth limit
    sudo tc qdisc add dev $INTERFACE handle ffff: ingress
    sudo tc filter add dev $INTERFACE parent ffff: protocol ip u32 match u32 0 0 action mirred egress redirect dev $IFB_INTERFACE
    sudo tc qdisc add dev $IFB_INTERFACE root tbf rate $RATE burst $BURST latency $LATENCY
    
    # Show new state
    echo "Traffic control rules applied. New state:"
    show_tc $INTERFACE
    show_tc $IFB_INTERFACE
}

function remove_tc {
    echo "Removing traffic control rules..."

    # Show current state
    show_tc $INTERFACE
    show_tc $IFB_INTERFACE
    
    # Remove outgoing bandwidth limit
    sudo tc qdisc del dev $INTERFACE root
    
    # Remove incoming bandwidth redirection and limit
    sudo tc qdisc del dev $INTERFACE ingress
    sudo tc qdisc del dev $IFB_INTERFACE root
    sudo ip link set $IFB_INTERFACE down
    sudo ip link delete $IFB_INTERFACE type ifb
    
    # Show new state
    echo "Traffic control rules removed. New state:"
    show_tc $INTERFACE
    show_tc $IFB_INTERFACE
}

case "$1" in
    apply)
        apply_tc
        ;;
    remove)
        remove_tc
        ;;
    show)
        echo "Showing current traffic control rules..."
        show_tc $INTERFACE
        show_tc $IFB_INTERFACE
        ;;
    *)
        echo "Usage: $0 {apply|remove|show}"
        exit 1
esac

exit 0
