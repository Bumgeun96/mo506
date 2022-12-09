from dqn import DQN
from loading_bdd100k import get_image_center,get_image_center1
from image_captioning import image_to_text
from data_load import get_image_and_center
import sys
import os
import pickle
import math

class Player():
    def __init__(self):
        self.dqn_obs_size = 11 #(hor_c,ver_c, x, y, x1, x2, y1, y2, a, h, v)
        self.dqn_act_size = 8 # number of discrete actions
        self.max_episode = 500
        self.max_epoch = 1500
        self.dqn_trainer = DQN(self.dqn_obs_size, self.dqn_act_size)
        self.image_to_text = image_to_text()
        self.done = False
        
    def image_truncation(self,images,center = [50,50],hor = 10,ver = 10):
        croppedImages = []
        horizontal,vertical = images.size
        croppedImage = images.crop((center[0]-hor,center[1]-ver,center[0]+hor,center[1]+ver))
        croppedImages.append(croppedImage)
        return croppedImages
        
    def reset(self):
        image,center_info = get_image_and_center()
        horizontal,vertical = image.size
        # reference_sentence = "a truck is driving down the road"
        reference_sentence = "a large group of trucks and cars are driving down a road"
        state = [int(horizontal/2),int(vertical/2),center_info[0],center_info[1],center_info[2],center_info[3],center_info[4],center_info[5],center_info[6],400,150]
        # state = [int(horizontal/2),int(vertical/2),int(center_info[0]),int(center_info[1])]
        self.done = False
        return state,image,reference_sentence
    
    def step(self,state,action,image):
        state_prime = state[:]
        step_size = 10
        horizontal,vertical = image.size
        center_x = state_prime[0]
        center_y = state_prime[1]
        
        # if action == 0: # +1 center horizontal
        #     if center_x > horizontal-step_size:
        #         state_prime[0] = state_prime[0]
        #     else:
        #         state_prime[0] += step_size
        # elif action == 1: # -1 center horizontal
        #     if center_x < step_size:
        #         state_prime[0] = state_prime[0]
        #     else:
        #         state_prime[0] -= step_size
        # elif action == 2: # +1 center vertical
        #     if center_y > vertical-step_size:
        #         state_prime[1] = state_prime[1]
        #     else:
        #         state_prime[1] += step_size
        # elif action == 3: # -1 center vertical
        #     if center_y < step_size:
        #         state_prime[1] = state_prime[1]
        #     else:
        #         state_prime[1] -= step_size
                
        if action == 0: # +1 horizontal
            if center_x + state_prime[-2] > horizontal-2:
                state_prime [-2] = state_prime[-2]
            else:
                state_prime[-2] = state_prime[-2] + 1
        elif action == 1: # -1 horizontal
            if state_prime[-2] < 11:
                state_prime[-2] = state_prime[-2]
            else:
                state_prime[-2] = state_prime[-2] - 1
        elif action == 2: # +1 vertical
            if center_y + state_prime[-1] > vertical-2:
                state_prime[-1] = state_prime[-1]
            else:
                state_prime[-1] = state_prime[-1] + 1
        elif action == 3: # -1 vertical
            if state_prime[-1] < 11:
                state_prime[-1] = state_prime[-1]
            else:
                state_prime[-1] = state_prime[-1] - 1
        elif action == 4: # +1 center horizontal
            if center_x+state_prime[-2] > horizontal-step_size:
                state_prime[0] = state_prime[0]
            else:
                state_prime[0] += step_size
        elif action == 5: # -1 center horizontal
            if center_x-state_prime[-2] < step_size:
                state_prime[0] = state_prime[0]
            else:
                state_prime[0] -= step_size
        elif action == 6: # +1 center vertical
            if center_y+state_prime[-1] > vertical-step_size:
                state_prime[1] = state_prime[1]
            else:
                state_prime[1] += step_size
        elif action == 7: # -1 center vertical
            if center_y-state_prime[-1] < step_size:
                state_prime[1] = state_prime[1]
            else:
                state_prime[1] -= step_size
        next_state = state_prime
        return next_state
    
    def get_reward(self,state,next_state,image,reference):
        current_image = self.image_truncation(image,[state[0],state[1]],hor=state[-2],ver=state[-1])
        # next_image = self.image_truncation(image,center = [next_state[0],next_state[1]],hor=next_state[-2],ver=next_state[-1])
        # current_text = self.image_to_text.image_captioning(current_image[0])
        # next_text = self.image_to_text.image_captioning(next_image[0])
        # print(current_text[0]['generated_text'])
        # print(reference)
        reward = 0
        if  math.sqrt((state[0]-int(state[2]))**2+(state[1]-int(state[3]))**2) < 14:
            current_text = self.image_to_text.image_captioning(current_image[0])
            reward += self.image_to_text.score(current_text[0]['generated_text'],reference)
        # if self.image_to_text.score(current_text[0]['generated_text'],reference) > 0.8:
        #     reward += 10
        # reward = self.image_to_text.score(current_text[0]['generated_text'],reference)
        # if reward > 0.8:
        #     reward += 10
        reward -= math.sqrt((state[0]-int(state[2]))**2+(state[1]-int(state[3]))**2)/100
        return reward

    def run(self):
        with open(os.path.dirname(os.path.realpath(__file__))+'/pickle/reward.pickle', 'wb') as f:
            pickle.dump(["center_x","center_y","hor","ver","reward"],f)
        for __ in range(self.max_episode):
            print("epi:",__+1)
            self.dqn_trainer.Epsilon()
            self.dqn_trainer.Loss()
            state, image,reference = self.reset()
            # image.show()
            reward_sum = 0
            for _ in range(self.max_epoch):
                print("==================================================")
                print(_+1,' step')
                action = self.dqn_trainer.select_action(state)
                next_state = self.step(state,action,image)
                reward = self.get_reward(state,next_state,image,reference)
                reward_sum += reward
                if _ == self.max_epoch-1:
                    self.done = True
                # print(state,next_state,action,reward)
                self.dqn_trainer.store_experience(state,next_state,action,reward,self.done)
                state = next_state
                self.dqn_trainer.update()
                self.dqn_trainer.save_checkpoint(_+1)
            with open(os.path.dirname(os.path.realpath(__file__))+'/pickle/reward.pickle', 'ab') as f:
                pickle.dump([state[0],state[1],state[-2],state[-1],reward],f)
            
    def test(self):
        state, image,reference = self.reset()
        for _ in range(self.max_epoch):
            action = self.dqn_trainer.select_action(state,test=True)
            state = self.step(state,action,image)
        current_image = self.image_truncation(image,[state[0],state[1]],hor=state[-2],ver=state[-1])
        current_image[0].show()
        current_text = self.image_to_text.image_captioning(current_image[0])
        print(current_text[0]['generated_text'])


if __name__ == '__main__':
    player = Player()
    # player.run()
    player.test()