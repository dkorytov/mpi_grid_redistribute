#!/usr/bin/env python2.7

from  __future__ import print_function, division


from mpi4py import  MPI
import numpy as np
import matplotlib.pyplot as plt

def peroidic_wrapping(data, box):
    return ((data%box)+box)%box

def mpi_grid_redistribute(data, pos, grid_topology, box_lengths, comm = comm, overload_lengths = None, peroidic=True):
    redist = MPIGridRedistributor(comm, grid_topology, box_lengths)
    return redist.redistriubte_by_position(data, pos, overload_lengths = overload_lengths, peroidic=peroidic)

class MPIGridRedistributor:
    def __init__(self, comm, grid_topology, box_length):
        """This class redistributes data by position on to a Cartesian
        grid of MPI ranks. Each MPI rank has all the data in it's own
        subvolume and any data in the overload region. 
        
        Parameters
        ----------
        comm : mpi4py communicator
          Instance of an mpi4py communicator. The class uses all mpi 
          calls throught it.
        
        grid_topology : array/list of ints
          The array sets the topology of the grid. The number elements is
          the dimensionality of the grid and the value of each element is 
          the number cells along that axis. 
        
        box_length : array/list 
          The bounding box for the data. Data with positions within the box
          will be redistributed on the grid of MPI ranks. Data with positions
          outside the box will be ignored. The variable must share the same 
          length as grid_topology. 

        """
        self.comm = comm
        self.grid_topology = np.array(grid_topology, dtype= np.int)
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        ranks_required = np.product(grid_topology)
        assert ranks_required <= self.size, "We have have {} ranks. The topology {} requires at least {} ranks".format(self.size, self.grid_topology, ranks_required)
        self.dim = len(self.grid_topology)
        self.box_length = np.array(box_length)
        assert len(self.dim) == len(self.box_length), "The number dimensions in grid_topoloby ({}), must be the same as in box_length ({})".format(len(self.grid_topology), len(self.box_length))
        # The uniform length of each grid cell in each dimension
        self.cell_length = np.zeros(self.dim)
        for d in range(0, self.dim):
            self.cell_length[d] = self.box_length[d]/self.grid_topology[d]
        # Create the multiplication factor from index to cell number
        self.cell_index_offset = np.zeros(self.dim, dtype=np.int)
        offset = 1
        for i in range(0, self.dim):
            j = (self.dim-1)-i
            self.cell_index_offset[j] =offset;
            offset *= grid_topology[j];
        # The ijk... index of the grid cell assigned to this rank
        self.rank_cell_index = self.get_indexes_from_cell_number(np.array([self.rank]))[0]
        self.rank_cell_limits = self.get_cell_limits_from_indexes(np.array([self.rank_cell_index]))[0]

    def get_cell_indexes_from_position(self, position, peroidic=True):
        cell_indexes = np.zeros(( len(position), self.dim), dtype=np.int)
            
        for d in range(0, self.dim):
            if peroidic:
                position[:,d] =peroidic_wrapping(position[:,d], self.box_length[d])
            data = (position[:,d]/self.box_length[d]*self.grid_topology[d]).astype(np.int)
            cell_indexes[:,d] = data
        return cell_indexes

    def get_cell_number_from_indexes(self, indexes, peroidic=True, check_range=True):
        cell_num = np.zeros(len(indexes), dtype=np.int)
        if not peroidic:
            for d in range(0, self.dim):
                cell_num += self.cell_index_offset[d]*indexes[:,d]
            if _check_range:
                for d in range(0, self.dim):
                    slct_outside = (indexes[:,d] > 0) & (indexes[:,d] < self.grid_topology[d])
                    cell_num[slct_outside] = -1
        else:
            for d in range(0, self.dim):
                cell_num += self.cell_index_offset[d] * (peroidic_wrapping(indexes[:, d], self.grid_topology[d]))
        return cell_num

    def get_cell_number_from_position(self, position, peroidic=True):
        indexes = self.get_cell_indexes_from_position(position, peroidic=peroidic)
        # these indexes are going to be within the range of allowed values
        return self.get_cell_number_from_indexes(indexes)
    
    def get_indexes_from_cell_number(self, cell_numbers):
        cell_indexes = np.zeros((len(cell_numbers), self.dim), dtype=int)
        for d in range(0, self.dim):
            cell_indexes[:,d] = cell_numbers / self.cell_index_offset[d]
            cell_numbers = cell_numbers % self.cell_index_offset[d]
        return cell_indexes

    def get_cell_limits_from_indexes(self, cell_indexes):
        """
        This function returns bounding box in each dimension for the
        given cell_indexes. 

        Paramters: 
        ----------
        cell_indexes: matrix_like
        
        """
        limits = np.zeros((len(cell_indexes),self.dim,2))
        for d in range(0, self.dim):
            limits[:,d,0] = cell_indexes[:, d]*self.cell_length[d]
            limits[:,d,1] = (cell_indexes[:, d]+1)*self.cell_length[d]
        return limits

    def redistribute_by_position(self, data, position, peroidic=True, 
                                 overload_length = None, return_positions = False):
        """

        Note: This function only uses
        the immediate neighbors. If the overload length extends beyond the
        immediate neighbors, it will not capture all data that would be in 
        the overloaded zone. 
        
        Paramters:
        ----------
        data: matrix_like/list_like
          The data to be redistributed. If it's a matrix, the frist
          axis is taken as interator axis. So the first element would be 
          data[0, :, :, ...], the second data[1, :, :, ...]
        
        position: matrix_like, shape=(N,d)
          The position of the data elements to be redistribtued. N is number
          of elements, d is dimension of the positions.

        peroidic: boolean, default = True
          if the position are peroidic or not. If they are, the position are 
          wrapped according to the box_lengths specified in the constructor. if
          not peroidic, then any data elements positioned outside of the box_lenghts
          are ignored and do not show up in the final result. 
   
        return_positions: boolean, default = false
        NOT IMPLMENTED
          This flag will make the function return the redistributed positions as well. 
          Typically the data will have the positions saved. But if memory is tight, we
          can avoid duplicating data. 
        
        
        Returns:
        --------
        local_data: same as data
          The data that has been assigned to this local rank and any overload
          data if the overload is specified
        
        """

        # TODO return positions
        rank_to_send = self.get_cell_number_from_position(position, peroidic=peroidic)
        assert np.min(rank_to_send) >= 0, "Trying to send to a negative rank. \nPosition is probably outside of box length"
        assert np.max(rank_to_send) < self.size, "Trying to send to a too rank number highre than max. \nPosition is probably outside of box length"
        local_data = redistribute_by_cell_number(self, data, rank_to_send)
        if overload_length is None:
            return local_data
        else:
            local_position = redistribute_by_cell_number(position, rank_to_send)
            overload_data = overload_buffer_matrix(local_data, local_position, overload_length)
            np.concatenate(local_data, overload_data, axis = 0)
        return 
        
    def redistribute_matrix_by_cell_number(self, data, rank_to_send):
        """
        This function redistributes the data by sending each data element
        to the rank specified. If the specified rank does not exist, that
        data will be ignored. 
        
        Parameters
        ----------
        data: list/array_like/matrix_like
          The local elements to the rank that should be overloaded. The first
          axis is taken as element index of data. The first element of the data
          would be data[0, :, :, ...], the second element data[1, :, :, ...], etc. 
        
        rank_to_send: array_like 
          Must have the same number of elements as data. These values are expected
          to be ints

        Returns
        -------
        
        local_data :  list/array_like/matrix_like
          The data elements that belong to the local rank. 
        """
        # TODO See if supports the actual data types listed. It works
        # for arrays/matrixes. But I don't know what happens if you
        # pass a list of strings or something
        send_buff = []
        for i in range(0, self.size):
            send_buff.append(data[rank_to_send==i])
        return np.concatenate(self.comm.alltoall(send_buff))
       
    def overload_buffer_matrix(self, data, position, overload_length, return_positions=false):
        """
        This function takes in the data that is divided already into cells
        and overloads a region around them. Note: This function only uses
        the immediate neighbors. If the overload length extends beyond the
        immediate neighbors, it will not capture all data that would be in 
        the overloaded zone. 

        Parameters
        ----------
        data: maxtrix_like
          The local elements to the rank that should be overloaded. The first
          axis is taken as ordering of the data. The first element of the data
          would be data[0, :, :, ...], the second data[1, :, :, ...]

        position: array_like
          The position of the elements that shouldbe be overloaded. Must have
          the shape of (N,d) where N is the number of elements in data and 
          d is the dimension of the 

        overload_length: array_like
          The distance to overload in each dimension. 
          
        Returns
        -------
        local_overload_data: list/array_like/matrix_like
          The data elements that belong the local rank's overload region. 

        return_positions: boolean, default = false
        NOT IMPLMENTED
          This flag will make the function return the redistributed positions as well. 
          Typically the data will have the positions saved. But if memory is tight, we
          can avoid duplicating data. 

         
        """

        # TODO: Check that data is within the bounding box for the mpi rank
        
        overload_shape = np.array(np.shape(data))
        overload_shape[0] = 0
        overload_data = np.zeros(overload_shape, dtype=data.dtype)
        overload_position = np.zeros((0,self.dim), dtype=position.dtype)
        # This is a hacky solution to having to send data twice around:
        # once for the "offical" data, and second for the position data. 
        overload_output = [overload_data, overload_position]
        for d in range(0,self.dim):
            offset_a = np.zeros(self.dim, dtype=np.int) # the rank to right
            offset_b = np.zeros(self.dim, dtype=np.int) # the rank to the left
            offset_a[d] = 1
            offset_b[d] = -1
            indexes_a = self.rank_cell_index + offset_a 
            indexes_b = self.rank_cell_index + offset_b
            cell_num_a = self.get_cell_number_from_indexes(np.array([indexes_a]))[0] # rank num to the right
            cell_num_b = self.get_cell_number_from_indexes(np.array([indexes_b]))[0] # rank num to the left
            
            # Send data twice. Once for the "offical" data and a second time for position data. 
            for datapos, overload_datapos, output_index in [(data, overload_data, 0), (position, overload_position, 1)]:
                # Here we will send data to the right and left rank in dimension d. 
                # The recieved data will be stored in an overload data buffer. 
                # When we are sending data, we send the rank local data (data that is within the
                # bounding box of the rank) as well as any data in the received overload region 
                # that need to be sent to another rank. This we reduce the total amount of 
                # communication that needs to be done. 

                # select the local data to send to the right(a) and left(a)
                cell_datapos_for_a = datapos[position[:, d] > (self.rank_cell_limits[d,1]-overload_length[d])]
                cell_datapos_for_b = datapos[position[:, d] < (self.rank_cell_limits[d,0]+overload_length[d])]
                # selected overloaded data
                buffer_datapos_for_a = overload_output[output_index][overload_output[1][:, d] > (self.rank_cell_limits[d,1]-overload_length[d])]
                buffer_datapos_for_b = overload_output[output_index][overload_output[1][:, d] < (self.rank_cell_limits[d,0]+overload_length[d])]

                # combine them into one array
                datapos_for_a = np.concatenate([cell_datapos_for_a, buffer_datapos_for_a], axis = 0)
                datapos_for_b = np.concatenate([cell_datapos_for_b, buffer_datapos_for_b], axis = 0)
                
                # we will send data to the right(a), and receive from the left(b)
                send_datapos_a_req = self.comm.isend(datapos_for_a, dest=cell_num_a, tag=0)
                recv_datapos_b_req = self.comm.irecv(source=cell_num_b, tag=0)
                
                # wait till we receive data from the left(b) and wait till we send the data to the right(a)
                datapos_from_b = recv_datapos_b_req.wait()
                send_datapos_a_req.wait()
                

                # we will send data to the left(b), and receive from the right(a)
                send_datapos_b_req = self.comm.isend(datapos_for_b, dest=cell_num_b, tag=0)
                recv_datapos_a_req = self.comm.irecv(source=cell_num_a, tag=0)
            
                # wait till we receive data from the right(a) and wait till we send the data to the left(b)
                datapos_from_a = recv_datapos_a_req.wait()
                send_datapos_b_req.wait()
                # Now we should have all the data we need!
                tmp = np.concatenate([overload_output[output_index], datapos_from_a, datapos_from_b])
                # if self.rank ==0:
                #     plt.figure()
                #     plt.plot(datapos[:, 0], datapos[:, 1], 'xk')
                #     plt.plot(datapos_from_a[:, 0], datapos_from_a[:,1], '.r')
                #     plt.plot(datapos_from_b[:, 0], datapos_from_b[:,1], '.b')
                #     plt.ylim([0,10])
                #     plt.xlim([0,10])
                #     plt.plot(tmp[:,0], tmp[:,1], 'g+')
                # store the data into the right buffer. Either the "official" data or the position data
                overload_output[output_index] = tmp

            # if self.rank == 0:
            #     plt.figure()
            #     plt.plot(data[:,0], data[:, 1], 'xk')
            #     plt.plot(overload_output[0][:,0], overload_output[0][:, 1], '.g')
            #     plt.ylim([0,10])
            #     plt.xlim([0,10])
            #     plt.show()
            # self.comm.Barrier()
        # return the "offical" overloaded data. The overloaded position data is thrown out. 
        return overload_output[0]


    
def redist():
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()
    data = np.ones(size*10)*rank*100
    
    data_size = 1000
    data = np.random.rand(data_size, 2)*10

    redist = Redistributor(comm, [np.sqrt(size),np.sqrt(size)], [10,10])
    # for i in range(0, size):
    #     if rank == i:
    #         print("{}/{}".format(rank, size))
    #         plt.figure()
    #         plt.plot(data[:, 0], data[:, 1], '.')
    #         plt.ylim([0,10])
    #         plt.xlim([0,10])
    #     comm.Barrier()
    num = redist.get_cell_number_from_position(data)
    data2 = redist.redistribute_matrix_by_cell_number(data, num)
    # if rank == 0:
    #     print(data2)
    # for i in range(0, size):
    #     if rank == i and rank==0:
    #         print("{}/{}".format(rank, size))
    #         plt.figure()
    #         plt.plot(data2[:, 0], data2[:, 1], '.')
    #         plt.ylim([0,10])
    #         plt.xlim([0,10])
    #     comm.Barrier()
    position = data2

    buffer_data = redist.overload_buffer_matrix(data2, position, [2, 2])

    data_local = redist.redistribute_matrix_by_position(data, data, overload_lenght =2)
    plt.show()

if __name__ == "__main__":
    redist();
