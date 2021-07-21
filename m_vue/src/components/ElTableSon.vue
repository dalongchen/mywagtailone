<template>
<div>
  <!-- flag === '1'为公告利好table 页面菜单stock  -->
  <div v-if="son_props.flag === '1'">
    <el-table   
      :data="son_props.data"
      style="width: 100%"
    >
      <el-table-column prop="notice_date" label="日期" sortable width="180"></el-table-column>
      <el-table-column prop="dis_time" label="公告" sortable width="200"></el-table-column>
      <el-table-column prop="col_type" label="类型" width="180"></el-table-column>
      <el-table-column prop="title" label="标题"></el-table-column>  
    </el-table>
  </div>
  <!-- flag === '2'为列标题传入的公用表,data数据为列表list  -->
  <div v-else-if="son_props.flag === '2'">
    <el-tag type="info">标签三</el-tag>
    <el-table
      :data="son_props.data"
      style="width: 100%"
      empty-text="null"
    >
      <el-table-column
          width="180" 
          v-for=" i in son_props.th"
          :key="i.id"
          :property="i.prop"
          :label="i.propName">
      </el-table-column> 
    </el-table>
  </div>
  <!-- flag === '3'为不需要表头,表头已经在data数据列表list里面传入  -->
  <div v-else-if="son_props.flag === '3'">
    <el-tag type="info">标签三2</el-tag>
    <el-table
      :data="son_props.data"
      style="width: 100%"
      empty-text="null"
      show-header=false
    >
      <el-table-column
          width="100">
      </el-table-column>
    </el-table>
  </div>
</div>
</template>
 
<script lang="ts">
import { defineComponent } from "vue";

export default defineComponent({
  name: "ElTableSon",
  props: {
    son_props: {},
  },
  
});
</script>