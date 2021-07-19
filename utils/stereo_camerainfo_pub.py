import rospy
import yaml

from sensor_msgs.msg import CameraInfo

class NerianStereoInfoPub():
    def __init__(self):
        info_file = rospy.get_param('~info_file','nerian_camera_info.yaml')
        with open(info_file) as f:
            info = yaml.load(f, Loader=yaml.SafeLoader)
        self.new_infos = self.get_infos(info)
        self.copy_left = rospy.get_param('~copy_left', False) # more generally, False
        in_topic_l = rospy.get_param('~in_topic_l','/nerian/left/camera_info_mono')
        in_topic_r = rospy.get_param('~in_topic_r','/nerian/right/camera_info_mono')
        out_topic_l = rospy.get_param('~out_topic_l','/nerian/left/camera_info')
        out_topic_r = rospy.get_param('~out_topic_r','/nerian/right/camera_info')
        self.sub_l = rospy.Subscriber(in_topic_l, CameraInfo, self.cb_l)
        self.sub_r = rospy.Subscriber(in_topic_r, CameraInfo, self.cb_r)
        self.pub_l = rospy.Publisher(out_topic_l, CameraInfo, queue_size=5)
        self.pub_r = rospy.Publisher(out_topic_r, CameraInfo, queue_size=5)

    def get_infos(self, info):
        # print('input infos',info)
        new_infos = {}
        new_infos['left'] = CameraInfo()
        new_infos['left'].width = info['size'][0]
        new_infos['left'].height = info['size'][1]
        new_infos['left'].distortion_model = "plumb_bob"
        new_infos['left'].D = info['D1']
        new_infos['left'].K = info['M1']
        new_infos['left'].R = info['R1']
        new_infos['left'].P = info['P1']
        
        new_infos['right'] = CameraInfo()
        new_infos['right'].width = info['size'][0]
        new_infos['right'].height = info['size'][1]
        new_infos['right'].distortion_model = "plumb_bob"
        new_infos['right'].D = info['D2']
        new_infos['right'].K = info['M2']
        new_infos['right'].R = info['R2']
        new_infos['right'].P = info['P2']
        return new_infos

    def cb_l(self, m):
        if self.copy_left:
            # self.new_infos['left'] = m
            pass
        else:
            self.new_infos['left'].header = m.header
        self.pub_l.publish(self.new_infos['left'])

    def cb_r(self, m):
        self.new_infos['right'].header = m.header
        self.new_infos['right'].header.frame_id = self.new_infos['left'].header.frame_id
        self.pub_r.publish(self.new_infos['right'])

if __name__ == "__main__":
    rospy.init_node('stereo_info_pub')
    n = NerianStereoInfoPub()
    rospy.spin() 