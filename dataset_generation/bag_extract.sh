#!/usr/bin/env bash

OPTIND=1

class='testclass'
overwrite=0
last_n=0
bag_dir="$HOME/data/rcta_objects/barrier"
sync="approx"
color="/camera/color/image_rect_color"
depth="/camera/aligned_depth_to_color/image_raw"
out=""
undistort=0
all_topics=()
all_types=()
topics=""
cam_info=""
sensor_name=""
sensor_ns=""
sensor_type=""
usage="$(basename "$0") [-h] [-b bag_dir (default '$bag_dir')] [-c class (default '$class')] -w overwrite sequence dirs, starting at seq 0. Use with caution. (no argument for this flag) [-l last_n (default 0. A value > 0 will process only the most recent n bag files)] [-s sync (default 'approx')] [-u undistort. No argument for this flag] [-o output_dir (default <bag_dir>/../)] [-t topics. Comma separated list of topics]"

while getopts "ho:wua:b:s:i:c:l:t:" opt; do # if a colon follows an option, the option comes with a argument
  case "$opt" in
    h)
      echo $usage
      exit 0
      ;;
    o)
      out=$OPTARG
      ;;
    w)
      overwrite=1
      ;;
    u)
      undistort=1
      ;;
    c)
      class=$OPTARG
      ;;
    b)
      bag_dir=$OPTARG
      ;;
    s)
      sync=$OPTARG
      ;;
    i)
      info=$OPTARG
      ;;
    t)
      IFS=',' topics=($OPTARG)
      ;;
    l)
      last_n=$OPTARG
  esac
done
shift $((OPTIND-1))
#[ "$1" = "--" ] && shift

if [[ -z $out ]]; then
  out="${bag_dir}/../"
fi

synchronize() {
  b=$1
  if [[ $sync == "approx" ]]; then
      ./synchronize_frames.py --dataset $b --topic_filter ${all_topics[@]} --type_filter ${all_types[@]} --approx
  else
      ./synchronize_frames.py --dataset $b --topic_filter ${all_topics[@]} --type_filter ${all_types[@]}
  fi
}

get_sensor_ns() {
    t=$1
    topic_stem=$(echo $t | sed 's|.*/||')
    sensor_ns=$(echo $t | sed 's|\(.*\)/.*|\1|')
    ns_stem=$(echo $sensor_ns | sed 's|.*/||')
    if [[ $topic_stem == "compressed" ]]; then
      sensor_ns=$(echo $sensor_ns | sed 's|\(.*\)/.*|\1|')
    elif [[ $ns_stem == "color" ]]; then
      sensor_ns=$(echo $sensor_ns | sed 's|\(.*\)/.*|\1|')
    fi
    sensor_name=$(echo $sensor_ns | sed 's|.*/||')
}

get_cam_info() {
    b=$1
    t=$2
    cam_info=""
    get_sensor_ns $t
    ns_topics=$(rosbag info $b | grep $sensor_ns)
    info=$(echo $ns_topics | grep "sensor_msgs/CameraInfo")
    if [[ $info ]]; then
      cam_info=$(echo $info | tr -s ' ' | cut -c2- | cut -d ' ' -f1)
    fi
}

get_topic_type() {
    b=$1
    t=$2
    bag_info=$(rosbag info $b | grep $t)
    if [[ $(echo $bag_info | grep "sensor_msgs/Image") ]]; then
      if [[ $(echo $t | grep "depth") ]]; then
        sensor_type="DepthImage"
      else
          sensor_type="Image"
      fi
    elif [[ $(echo $bag_info | grep "sensor_msgs/CameraInfo") ]]; then
      sensor_type="CameraInfo"
    elif [[ $(echo $bag_info | grep "sensor_msgs/PointCloud2") ]]; then
      sensor_type="PointCloud2"
    else
        echo "Unknown type for topic $t"
    fi

}

get_topic_types() {
    b=$1
    for t in ${topics[@]}; do
        get_topic_type $b $t
        if [[ $sensor_type == "Image" || $sensor_type == "DepthImage" ]]; then
          all_topics+=($t)
          all_types+=("sensor_msgs/Image")
          get_cam_info $b $t
          if [[ -n $cam_info ]]; then
            all_topics+=($cam_info)
            all_types+=("sensor_msgs/CameraInfo")
          else
              echo "Unable to find CameraInfo for $t"
          fi
        elif [[ $sensor_type == "PointCloud2" ]]; then
          all_topics+=($t)
          all_types+=("sensor_msgs/PointCloud2")
        fi
    done
}

extract() {
    b=$1
    i=$2
    for t in ${all_topics[@]}; do
        get_topic_type $b $t
        get_sensor_ns $t
        if [[ $sensor_type == "DepthImage" ]]; then
          ./extract_images.py --bag_file $b --undistort $undistort --output_dir $out/$class/$i/$sensor_name/ --image_topic $t --image_info_topic $cam_info --image_enc mono16
        elif [[ $sensor_type == "Image" ]]; then
             ./extract_images.py --bag_file $b --undistort $undistort --output_dir $out/$class/$i/$sensor_name/ --image_topic $t --image_info_topic $cam_info --image_enc bgr8 --file_format jpg
        elif [[ $sensor_type == "PointCloud2" ]]; then
             ./extract_ply.py --bag_file $b --output_dir $out/$class/$i/$sensor_name --pcd_topic $t
        fi
    done
}

if [[ $last_n == 0 ]]; then
  readarray -t all_bags<<<"$(ls -hardt $bag_dir/*bag | grep -v synced.bag)"
else
  readarray -t all_bags<<<"$(ls -hardt $bag_dir/*bag | grep -v synced.bag | tail -n $last_n)"
fi

echo "Num bags ${#all_bags[@]}"
if [[ ! -f $out/$class ]]; then
    mkdir -p $out/$class
fi

if [[ $overwrite == 1 ]]; then
    seq_id=0
else
    seq_id=`find $out/$class -maxdepth 1 -type d -name "0*" | wc -l`
fi

echo "seq id $seq_id"

if [[ ! $(rostopic list) ]]; then
    roscore&
fi

for b in "${all_bags[@]}"; do
    echo $b
    seq_dir=$(printf "%05d" $seq_id)
    echo "seq dir $seq_dir"
    bag_synced="$(dirname $b)/$(basename $b ".bag")_synced.bag"
    echo "output $out/$class/$seq_dir/rgb"
    num_sensors="${#topics[@]}"
    get_topic_types $b
    echo "All topics ${topics[@]}"
    echo "First topic ${topics[0]}"
    if [[ $num_sensors -gt 1 ]]; then
      echo "syncing $bag_synced"
      synchronize $b
      extract $bag_synced $seq_dir
    else
      extract $b $seq_dir
    fi
    if [[ $(ls $out/$class/$seq_dir/$sensor_name | wc -l) != "0" ]]; then
      seq_id=$(($seq_id+1))
    fi
done
